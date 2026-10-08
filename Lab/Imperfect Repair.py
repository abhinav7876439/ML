import numpy as np
import pandas as pd
# Try to import markovianbandit; if not available, skip package comparison
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; skip package comparison.")


# ----------------------------------------------------------------------
# 1. Extended MDP builder — IMPERFECT REPAIR (two-component state (k,j))
# ----------------------------------------------------------------------
def build_extended_mdp_imperfect(N, p, q, c, K, c_switch):
    """
    State space: active family (k,1), k=0,...,N  and passive family (k,0), k=0,...,N.
    Indexing convention:
        idx = k            for state (k,1),  k = 0,...,N   (first N+1 slots)
        idx = (N+1) + k     for state (k,0),  k = 0,...,N   (next N+1 slots)

    Parameters
    ----------
    N : int
        Max deterioration level tracked (truncation level).
    p : array-like, length N+1
        p[k] = P(survive | level k, passive action), used for k = 0,...,N-1.
        (p[N] is irrelevant since (N,.) is the truncation boundary — forced
         to breakdown under passive action, matching the boundary handling
         in your original 1-D code.)
    q : list of arrays
        q[k] is a probability vector of length k+1: q[k][i] = P(repair outcome
        lands at level i | active action applied at level k), for i = 0,...,k.
        Same q_k(.) is used regardless of whether the repairman was already
        present (state (k,1)) or freshly called in (state (k,0)) — matches
        the diagram, where the *outcome* distribution is identical, only the
        *cost* differs.
    c : float
        Cost of active action when repairman already present (from (k,1)).
    K : float
        Breakdown cost.
    c_switch : float
        Extra cost of active action when repairman must be newly assigned
        (from (k,0)).

    Returns
    -------
    P0, P1 : (2(N+1) x 2(N+1)) transition matrices for passive / active action
    R0, R1 : reward vectors (negative costs) for passive / active action
    state_labels : list of state name strings, same order as the matrix indices
    """
    num_active = N + 1
    num_states = 2 * (N + 1)

    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)

    def act_idx(k):
        return k
    def pas_idx(k):
        return num_active + k

    state_labels = [f"({k},1)" for k in range(N + 1)] + [f"({k},0)" for k in range(N + 1)]

    # sanity check on repair-quality distributions
    for k in range(N + 1):
        qk = np.asarray(q[k], dtype=float)
        assert qk.shape[0] == k + 1, f"q[{k}] must have length {k+1}"
        assert np.isclose(qk.sum(), 1.0), f"q[{k}] must sum to 1 (got {qk.sum()})"

    # ---------------- ACTIVE ACTION (a = 1) — repair, Diagram 1 ----------
    for k in range(N + 1):
        qk = np.asarray(q[k], dtype=float)

        # from active family (k,1): repairman already present, cost c
        s = act_idx(k)
        for i in range(k + 1):
            P1[s, act_idx(i)] += qk[i]
        R1[s] = -c

        # from passive family (k,0): repairman newly assigned, cost c + c_switch
        s = pas_idx(k)
        for i in range(k + 1):
            P1[s, act_idx(i)] += qk[i]
        R1[s] = -(c + c_switch)

    # ---------------- PASSIVE ACTION (a = 0) — survive/breakdown, Diagram 2
    for k in range(N + 1):
        for s in (act_idx(k), pas_idx(k)):
            if k < N:
                P0[s, pas_idx(k + 1)] = p[k]          # survive -> (k+1,0), cost 0
                P0[s, pas_idx(0)] = 1 - p[k]           # breakdown -> (0,0), cost K
                R0[s] = -(K * (1 - p[k]))
            else:
                # truncation boundary: forced breakdown, as in the original code
                P0[s, pas_idx(0)] = 1.0
                R0[s] = -K

    return P0, P1, R0, R1, state_labels


def print_transition_matrix(P, state_labels, action_name):
    print("=" * 70)
    print(f"Transition matrix for action {action_name}")
    print("=" * 70)
    df = pd.DataFrame(P, index=state_labels, columns=state_labels)
    print(df)


def print_reward_vector(R, state_labels, action_name):
    print("=" * 70)
    print(f"Reward vector for action {action_name}")
    print("=" * 70)
    df = pd.DataFrame(R, index=state_labels, columns=[action_name])
    print(df)


# ----------------------------------------------------------------------
# 2. Convenience: a default repair-quality distribution q_k(.)
# ----------------------------------------------------------------------
def make_geometric_repair_dist(N, r=0.5):
    """
    Builds a default q[k] for k=0,...,N: geometric-decaying mass on the
    'best' outcomes i=0,1,2,..., with whatever mass is left over dumped on
    i=k (repair completely fails, machine unchanged) so it sums to 1.
    Purely illustrative — replace with fitted/estimated values as needed.
    """
    q = []
    for k in range(N + 1):
        if k == 0:
            q.append(np.array([1.0]))   # level 0: active action is a trivial full repair
            continue
        raw = np.array([r ** i for i in range(k)])   # mass for i = 0,...,k-1
        raw = raw / raw.sum() * 0.8                   # keep 80% mass for "some repair happens"
        qk = np.append(raw, 1 - raw.sum())             # remaining 20% -> i = k (repair fails)
        q.append(qk)
    return q




# ----------------------------------------------------------------------
# 2. Discounted solvers (VI, PI)
# ----------------------------------------------------------------------
def solve_vi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=10000):
    S = len(R0)
    V = np.zeros(S)
    for _ in range(max_iter):
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - w) + beta * (P1 @ V)
        V_new = np.maximum(Q0, Q1)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    Q0 = R0 + beta * (P0 @ V)
    Q1 = (R1 - w) + beta * (P1 @ V)
    return V, Q0, Q1

def policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta):
    S = len(pi)
    P_pi = np.array([P0[s] if pi[s] == 0 else P1[s] for s in range(S)])
    r_pi = np.array([R0[s] if pi[s] == 0 else R1[s] - w for s in range(S)])
    I = np.eye(S)
    V = np.linalg.solve(I - beta * P_pi, r_pi)
    return V

def solve_pi_discounted(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=1000):
    S = len(R0)
    pi = np.zeros(S, dtype=int)
    for _ in range(max_iter):
        V = policy_evaluation_discounted(pi, P0, P1, R0, R1, w, beta)
        Q0 = R0 + beta * (P0 @ V)
        Q1 = (R1 - w) + beta * (P1 @ V)
        pi_new = (Q1 > Q0).astype(int)
        if np.all(pi_new == pi):
            break
        pi = pi_new
    Q0 = R0 + beta * (P0 @ V)
    Q1 = (R1 - w) + beta * (P1 @ V)
    return V, Q0, Q1

def whittle_index_discounted(state, solver, beta, P0, P1, R0, R1,
                             w_min=-1000.0, w_max=1000.0,
                             tol_w=1e-5, tol_mdp=1e-8, verbose=False):
    solve_f = solve_vi_discounted if solver == 'vi' else solve_pi_discounted
    def f(w):
        _, Q0, Q1 = solve_f(P0, P1, R0, R1, w, beta, tol_mdp)
        return Q0[state] - Q1[state]
    f_min = f(w_min)
    f_max = f(w_max)
    expand = 1.5
    while f_min * f_max > 0:
        if abs(f_min) < abs(f_max):
            w_min -= expand * abs(w_min - w_max) if abs(w_min - w_max) > 0 else 10.0
            f_min = f(w_min)
        else:
            w_max += expand * abs(w_max - w_min)
            f_max = f(w_max)
        if abs(w_min) > 1e6 or abs(w_max) > 1e6:
            raise ValueError("Cannot bracket root")
    for _ in range(160):
        w_mid = (w_min + w_max) / 2.0
        f_mid = f(w_mid)
        if abs(f_mid) < tol_w:
            return w_mid
        if f_mid * f_min > 0:
            w_min = w_mid
            f_min = f_mid
        else:
            w_max = w_mid
    return (w_min + w_max) / 2.0


# ----------------------------------------------------------------------
# Compute Discounted Indices
# ----------------------------------------------------------------------
def compute_indices_discounted(R0, R1, P0, P1, beta, problem_name, state_labels=None):
    """Compute Whittle indices using discounted cost methods."""
    print(f"\n=== {problem_name} ===")
    results = {}
    
    for method in ['vi', 'pi']:
        print(f"\n--- {method.upper()} (Discounted, β={beta}) ---")
        indices = []
        for s in range(len(R0)):
            try:
                lam = whittle_index_discounted(state=s, solver=method, beta=beta,
                                               w_min=-1000.0, w_max=1000.0,
                                               tol_w=1e-5, tol_mdp=1e-8,
                                               P0=P0, P1=P1, R0=R0, R1=R1,
                                               verbose=False)
                indices.append(lam)
                if state_labels:
                    print(f"State {state_labels[s]:8s}: λ = {lam:8.4f}")
                else:
                    print(f"State {s+1}: λ = {lam:.10f}")
            except Exception as e:
                print(f"State {s+1}: Failed - {e}")
                indices.append(np.nan)
        results[method] = indices
    return results  

# ----------------------------------------------------------------------
# 3. Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    N = 4
    #p = [0.9, 0.85, 0.8, 0.75, 0.7]      # p[k], k = 0,...,N
    np.random.seed(42)
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]   # decreasing survival probabilities
    p_full = np.append(p, 0.0)                       # terminal zero for the extended MDP
    q = make_geometric_repair_dist(N, r=0.5)
    #c, K, c_switch = 1.0, 20.0, 0.5
    c, K = 5.0, 500.0
    c_switch = 600.0
    beta = 0.9999

    P0, P1, R0, R1, labels = build_extended_mdp_imperfect(N, p, q, c, K, c_switch)
    P0_ext, P1_ext, R0_ext, R1_ext, state_labels_ext = P0, P1, R0, R1, labels

    print_transition_matrix(P0, labels, "Passive (a=0)")
    print_transition_matrix(P1, labels, "Active (a=1)")
    print_reward_vector(R0, labels, "Passive (a=0)")
    print_reward_vector(R1, labels, "Active (a=1)")



    print("\n--- Numeric indices (Policy Iteration bisection) ---")
    numeric_indices = []
    for s in range(len(R0_ext)):
        lam = whittle_index_discounted(state=s, solver='pi', beta=beta,
                                       w_min=-100.0, w_max=1000.0,
                                       tol_w=1e-6, tol_mdp=1e-8,
                                       P0=P0_ext, P1=P1_ext, R0=R0_ext, R1=R1_ext)
        numeric_indices.append(lam)
        print(f"State {state_labels_ext[s]:8s}: λ = {lam:8.4f}")

    print("\n--- Compute indices using discounted methods ---")
    discounted_indices = compute_indices_discounted(R0_ext, R1_ext, P0_ext, P1_ext, beta, "Extended MDP (Imperfect Repair)", state_labels_ext)
    print("\nDiscounted indices (VI):", discounted_indices['vi'])
    print("Discounted indices (PI):", discounted_indices['pi']) 


    
    # ----------------------------------------------------------------------
    # Package Comparison for Extended MDP
    # ----------------------------------------------------------------------
    if PKG_AVAILABLE:
        print("\n" + "="*70)
        print("PACKAGE COMPARISON FOR EXTENDED MDP")
        print("="*70)
        
        # Build the extended MDP model
        model_ext = bandit.restless_bandit_from_P0P1_R0R1(P0_ext, P1_ext, R0_ext, R1_ext)
        
        # Discounted indices from package
        pkg_ext_disc = model_ext.whittle_indices(discount=beta)
        
        print("\nPackage discounted indices (β = {}):".format(beta))
        for s, (label, idx) in enumerate(zip(state_labels_ext, pkg_ext_disc)):
            print(f"  {label:8s}: λ = {idx:.6f}")
        
        # Average cost indices from package
        pkg_ext_avg = model_ext.whittle_indices()
        
        print("\nPackage average cost indices:")
        for s, (label, idx) in enumerate(zip(state_labels_ext, pkg_ext_avg)):
            print(f"  {label:8s}: λ = {idx:.6f}")


        # ----------------------------------------------------------------------
        # Summary Comparison Tables
        # ----------------------------------------------------------------------
        print("\n" + "="*70)
        print("SUMMARY COMPARISON - EXTENDED MDP")
        print("="*70)
        
        # Create DataFrame for discounted indices
        df_disc = pd.DataFrame({
            'State': state_labels_ext,
            'VI (Disc)': discounted_indices['vi'],
            'PI (Disc)': discounted_indices['pi'],
            #'Closed Form (Disc)': np.append(W_active, W_passive),
            'Package (Disc)': pkg_ext_disc if PKG_AVAILABLE else np.nan
        })
        print("\nDiscounted Indices:")
        print(df_disc)

