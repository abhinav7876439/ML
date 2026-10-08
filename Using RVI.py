import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import bisect
import networkx as nx
from scipy.optimize import brentq

# Try to import markovianbandit; if not available, skip package comparison
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; skip package comparison.")

# ------------------------------------------------------------
# 1. Model definition (P0, P1, R0, R1) for the extended state space
# ------------------------------------------------------------
def build_mdp(N, p, C, K, c_switch=0):
    """
    Build the extended MDP for a single machine with states (x,a_{-1}).
    Returns:
        P0, P1: transition matrices for passive (a=0) and active (a=1)
        R0, R1: immediate cost vectors for passive and active (negative costs)
    """
    n_states = (N+1)*2
    P0 = np.zeros((n_states, n_states))
    P1 = np.zeros((n_states, n_states))
    R0 = np.zeros(n_states)
    R1 = np.zeros(n_states)


    # Active/Passive offset
    # Indices 0 to N: Passive family (0,0) to (N,0)
    # Indices N+1 to 2N+1: Active family (0,1) to (N,1)

    for x in range(N+1):
        idx_p = x               # (x,0), Passive: 0 to N
        idx_a = x + (N+1)       # (x,1), Active: N+1 to 2N+1

        
        # --- Passive Action Dynamics (P0) ---
        # Catastrophic Breakdown (1-px) -> leads to (0,0), index 0
        P0[idx_p, 0] += (1 - p[x])
        P0[idx_a, 0] += (1 - p[x])
        
        # Deterioration (p_x)-> (x+1,0) if possible
        if x < N:
            P0[idx_p, x+1] += p[x]
            P0[idx_a, x+1] += p[x]
        else:
            # At last state, treat deterioration as failure (or stay)
            # Rule: Last state only has catastrophic failure, failure with prob 1 (p[N]=0)
            P0[idx_p, 0] += p[x]
            P0[idx_a, 0] += p[x]


        # Cost for passive: breakdown cost K * (1-p_x)
        R0[idx_p] = -K * (1 - p[x])
        R0[idx_a] = -K * (1 - p[x])

        # Active action (a=1) Dynamics (P1) -> always go to (0,1) i.e. index N+1
        P1[idx_p, N+1] = 1.0
        P1[idx_a, N+1] = 1.0
        R1[idx_p] = -(C + c_switch)   # switching cost for passive family
        R1[idx_a] = -C                 # no switching cost if already active

    return P0, P1, R0, R1

# ------------------------------------------------------------
# 2. Discounted value iteration for a given subsidy w
# ------------------------------------------------------------

# def discounted_value_iter(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=20000):
#     """
#     Solve the discounted Bellman equation for the w-subsidy problem.
#     Returns the value function V, the optimal action, and the number of iterations until convergence.
#     """
#     n = len(R0)
#     R1_mod = R1 - w
#     V = np.zeros(n)
#     converged_iter = max_iter
#     for it in range(max_iter):
#         V_old = V.copy()
#         Q0 = R0 + beta * (P0 @ V_old)
#         Q1 = R1_mod + beta * (P1 @ V_old)
#         V = np.minimum(Q0, Q1)          # no normalisation!
#         if np.max(np.abs(V - V_old)) < tol:
#             converged_iter = it + 1
#             break

#     Q0 = R0 + beta * (P0 @ V)
#     Q1 = R1_mod + beta * (P1 @ V)
#     opt_action = (Q1 < Q0).astype(int)
#     return V, opt_action, converged_iter


# def discounted_value_iteration(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=20000):
#     n = len(R0)
#     R1_mod = R1 - w
#     V = np.zeros(n)
#     for it in range(max_iter):
#         V_old = V
#         Q0 = R0 + beta * (P0 @ V_old)
#         Q1 = R1_mod + beta * (P1 @ V_old)
#         V = np.minimum(Q0, Q1)
#         if np.linalg.norm(V - V_old, ord=np.inf) < tol:
#             break
#     return V   


# def discounted_whittle_bisection(state_idx, P0, P1, R0, R1, beta, w_min, w_max, tol=1e-6):
#     def advantage(w):
#         V, _, iters = discounted_value_iteration(P0, P1, R0, R1, w, beta)
#         print(f"State {state_idx}, w={w:.4f}, converged in {iters} iterations")
#         Q_pass = R0[state_idx] + beta * (P0[state_idx, :] @ V)
#         Q_act = (R1[state_idx] - w) + beta * (P1[state_idx, :] @ V)
#         return Q_pass - Q_act

#     # Expand bounds if needed
#     w_low, w_high = w_min, w_max
#     f_low = advantage(w_low)
#     f_high = advantage(w_high)
#     while f_low * f_high > 0 and (w_high - w_low) < 1e6:
#         if f_low > 0:
#             w_low -= (w_high - w_low)
#         else:
#             w_high += (w_high - w_low)
#         f_low = advantage(w_low)
#         f_high = advantage(w_high)

#     try:
#         return brentq(advantage, w_low, w_high, xtol=tol)
#     except ValueError:
#         # fallback: choose the bound where advantage is closer to zero
#         return w_low if abs(f_low) < abs(f_high) else w_high


# def compute_all_indices_discounted(N, p, C, K, c_switch, beta, w_range=(-100, 500)):
#     P0, P1, R0, R1 = build_mdp(N, p, C, K, c_switch)
#     indices = np.zeros(2*(N+1))
#     for s in range(2*(N+1)):
#         indices[s] = discounted_whittle_bisection(s, P0, P1, R0, R1, beta, w_range[0], w_range[1])
#     return indices

# ------------------------------------------------------------
# 3. Average cost via Relative Value Iteration (RVI)
# ------------------------------------------------------------



# def average_cost_rvi(P0, P1, R0, R1, w, ref_state=0, tol=1e-8, max_iter=30000):
#     """
#     Solve average cost Bellman equation for subsidy w using Relative Value Iteration.
#     Returns (gain, relative values h) with h(ref_state)=0.
#     """
#     n = len(R0)
#     R1_mod = R1 - w
#     h = np.zeros(n)
#     converged_iter = max_iter
#     for it in range(max_iter):
#         h_old = h.copy()
#         Q0 = R0 + P0 @ h_old
#         Q1 = R1_mod + P1 @ h_old
#         h_new = np.minimum(Q0, Q1)
#         # Subtract the value at the reference state
#         offset = h_new[ref_state]
#         h_new = h_new - offset
#         if np.max(np.abs(h_new - h_old)) < tol:
#             converged_iter = it + 1
#             break
#         h = h_new
#     # Final gain estimate: λ = limiting value of h_n(ref_state) = offset
#     # But offset is already subtracted, so gain is stored separately.
#     # Recompute Q to get best actions.
#     Q0 = R0 + P0 @ h
#     Q1 = R1_mod + P1 @ h
#     return h, (Q1 < Q0), converged_iter   # we don't actually need λ for the advantage



# def find_bracketing_interval(advantage_func, w_init_low, w_init_high, max_expand=10, expand_factor=2.0):
#     low, high = w_init_low, w_init_high
#     f_low = advantage_func(low)
#     f_high = advantage_func(high)
#     for _ in range(max_expand):
#         if f_low * f_high <= 0:
#             return low, high
#         if abs(f_low) < abs(f_high):
#             low = low - expand_factor * (high - low)
#             f_low = advantage_func(low)
#         else:
#             high = high + expand_factor * (high - low)
#             f_high = advantage_func(high)
#     raise ValueError("Could not find sign change after expansion")




# def discounted_value_iteration(P0, P1, R0, R1, w, beta, tol=1e-8, max_iter=20000):
#     n = len(R0)
#     R1_mod = R1 - w
#     V = np.zeros(n)
#     for _ in range(max_iter):
#         V_old = V
#         Q0 = R0 + beta * (P0 @ V_old)
#         Q1 = R1_mod + beta * (P1 @ V_old)
#         V = np.minimum(Q0, Q1)
#         if np.max(np.abs(V - V_old)) < tol:
#             break
#     return V

# ---------- Average cost RVI with reference state ----------
# def average_rvi(P0, P1, R0, R1, w, ref_state=0, tol=1e-8, max_iter=30000):
#     n = len(R0)
#     R1_mod = R1 - w
#     h = np.zeros(n)
#     for _ in range(max_iter):
#         h_old = h
#         Q0 = R0 + P0 @ h_old
#         Q1 = R1_mod + P1 @ h_old
#         h_new = np.minimum(Q0, Q1)
#         h_new -= h_new[ref_state]      # centre on reference state
#         if np.max(np.abs(h_new - h_old)) < tol:
#             break
#         h = h_new
#     return h   # relative values with h(ref_state)=0

# ---------- Generic bisection with dynamic bounds ----------
# def whittle_index(state_idx, P0, P1, R0, R1, use_discount, beta=None,
#                   w_init=0.0, w_span=500.0, tol=1e-6):
#     """
#     use_discount: True -> discounted, False -> average cost.
#     """
#     def advantage(w):
#         if use_discount:
#             V = discounted_value_iteration(P0, P1, R0, R1, w, beta)
#             Q_pass = R0[state_idx] + beta * (P0[state_idx, :] @ V)
#             Q_act = (R1[state_idx] - w) + beta * (P1[state_idx, :] @ V)
#         else:
#             h = average_rvi(P0, P1, R0, R1, w)
#             Q_pass = R0[state_idx] + P0[state_idx, :] @ h
#             Q_act = (R1[state_idx] - w) + P1[state_idx, :] @ h
#         return Q_pass - Q_act

#     # Expand bounds until sign change
#     w_low, w_high = w_init - w_span, w_init + w_span
#     f_low = advantage(w_low)
#     f_high = advantage(w_high)
#     while f_low * f_high > 0 and (w_high - w_low) < 1e6:
#         if f_low > 0:
#             w_low -= (w_high - w_low)
#         else:
#             w_high += (w_high - w_low)
#         f_low = advantage(w_low)
#         f_high = advantage(w_high)

#     try:
#         return brentq(advantage, w_low, w_high, xtol=tol)
#     except ValueError:
#         return w_low if abs(f_low) < abs(f_high) else w_high


# def compute_all_indices_average(N, p, C, K, c_switch, w_range=(-100, 500)):
#     P0, P1, R0, R1 = build_mdp(N, p, C, K, c_switch)
#     indices = np.zeros(2*(N+1))
#     for s in range(2*(N+1)):
#         indices[s] = whittle_index(s, P0, P1, R0, R1, use_discount=False, w_init=0.0, w_span=500.0)
#     return indices

# ------------------------------------------------------------
# 4. Closed-form expressions (derived earlier)
# ------------------------------------------------------------
def compute_G_H(p, beta):
    """ Computes G(k) and H(k) based on transition probabilities. """
    N = len(p)
    H = np.zeros(N)
    G = np.zeros(N+1)
    for k in range(N):
        prod = 1.0
        for i in range(k):
            prod *= p[i]
        H[k] = beta**(k+1) * prod
        G[k+1] = G[k] + (1 - p[k]) * H[k]
    return G, H



def closed_form_paper_indices_discounted(p, C, K, beta):
    """Return array W(x,1) for x=0..N (active state indices)."""
    G, H = compute_G_H(p, beta)
    N = len(p)
    W = np.zeros(N)
    for k in range(N):
        num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
        den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
        W[k] = K * num / den - C
    return W





def compute_T_k_avg(p):
    """
    Computes expected cycle times via recurrence:
    T[0] = 1
    T[k] = 1 + p[k-1] * T[k-1]
    """
    num_states = len(p)
    T = np.zeros(num_states+1)
    T[0] = 1.0
    for k in range(1, num_states+1):
        T[k] = 1 + p[k-1] * T[k-1]
    return T

def closed_form_paper_indices_average(p, C, K):
    """
    Closed-form average cost Whittle indices (Corollary 2).
    W(k) = K * (1 - p[k] / (T[k+1] - p[k] * T[k])) - C
    """
    N = len(p)
    T = compute_T_k_avg(p)
    W = np.zeros(N)
    for k in range(N):
        denom = T[k+1] - p[k] * T[k]
        if denom > 0:
            W[k] = K * (1 - p[k] / denom) - C
        else:
            W[k] = K - C
    return W

# N = 7
# # --- Visualization ---
# def plot_transitions(P0, P1):
#     G = nx.MultiDiGraph()
#     labels = {i: f"({i%(N+1)}, {'A' if i<=N else 'P'})" for i in range(len(P0))}
    
#     for i in range(len(P0)):
#         for j in range(len(P0)):
#             if P0[i, j] > 0: G.add_edge(i, j, color='blue', label='b')
#             if P1[i, j] > 0: G.add_edge(i, j, color='red', label='a')
            
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, labels=labels, with_labels=True, node_color='lightblue', node_size=1500)
#     plt.title("Transition Diagram (Red=Active, Blue=Passive)")
#     plt.show()







# ------------------------------------------------------------
# 5. Main: run all methods and compare
# ------------------------------------------------------------
def main():
    np.random.seed(42)
    N = 4 
    # Generate decreasing survival probabilities (p_N=0)
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]
    p = np.append(p, 0.0)
    print("Survival probabilities p_x:", np.round(p, 3))

    # Costs
    C = 5.0
    K = 500.0
    c_switch = 0.0   # no switching for now
    beta = 0.95

    # 1. Closed-form indices
    W_closed_disc = closed_form_paper_indices_discounted(p, C, K, beta)
    # For average cost
    W_closed_avg = closed_form_paper_indices_average(p, C, K)

    # # 2. RVI+bisection indices
    # print("\nComputing discounted indices via RVI+bisection...")
    # ind_disc_rvi = compute_all_indices_discounted(N, p, C, K, c_switch, beta, w_range=(-50, 300))
    # # Only active states (x,1) correspond to indices N+1 ... 2N+1
    # ind_disc_rvi_active = ind_disc_rvi[N+1:2*(N+1)]

    # print("Computing average cost indices via RVI+bisection...")
    # ind_avg_rvi = compute_all_indices_average(N, p, C, K, c_switch, w_range=(-50, 300))
    # ind_avg_rvi_active = ind_avg_rvi[N+1:2*(N+1)]

    # 3. Package comparison (if available)
    if PKG_AVAILABLE:
        P0, P1, R0, R1 = build_mdp(N, p, C, K, c_switch)
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
        pkg_disc = model.whittle_indices(discount=beta)
        pkg_avg = model.whittle_indices(discount=0.0)
        pkg_disc_active = pkg_disc[N+1:2*(N+1)]
        pkg_avg_active = pkg_avg[N+1:2*(N+1)]
    else:
        pkg_disc_active = pkg_avg_active = None

    # # 4. Print comparison table
    # print("\n" + "="*80)
    # print("Discounted indices (beta = {})".format(beta))
    # print("-"*80)
    # header = f"{'x':>3} | {'Closed-form':>12} | {'RVI+bisect':>12} | {'Package':>12} | {'Diff (RVI - closed)':>20}"
    # print(header)
    # print("-"*80)
    # for x in range(N+1):
    #     closed = W_closed_disc[x]
    #     rvi_val = ind_disc_rvi_active[x]
    #     pkg_val = pkg_disc_active[x] if pkg_disc_active is not None else np.nan
    #     diff = rvi_val - closed
    #     print(f"{x:3} | {closed:12.4f} | {rvi_val:12.4f} | {pkg_val:12.4f} | {diff:20.4f}")

    # print("\n" + "="*80)
    # print("Average cost indices")
    # print("-"*80)
    # header = f"{'x':>3} | {'Closed-form':>12} | {'RVI+bisect':>12} | {'Package':>12} | {'Diff (RVI - closed)':>20}"
    # print(header)
    # print("-"*80)
    # for x in range(N+1):
    #     closed = W_closed_avg[x]
    #     rvi_val = ind_avg_rvi_active[x]
    #     pkg_val = pkg_avg_active[x] if pkg_avg_active is not None else np.nan
    #     diff = rvi_val - closed
    #     print(f"{x:3} | {closed:12.4f} | {rvi_val:12.4f} | {pkg_val:12.4f} | {diff:20.4f}")

    # # Optional plot
    # plt.figure(figsize=(8,5))
    # plt.plot(range(N+1), W_closed_disc, 'o-', label='Closed-form (disc)')
    # plt.plot(range(N+1), ind_disc_rvi_active, 's--', label='RVI+bisect (disc)')
    # if pkg_disc_active is not None:
    #     plt.plot(range(N+1), pkg_disc_active, '^:', label='markovianbandit (disc)')
    # plt.xlabel('Deterioration state x')
    # plt.ylabel('Whittle index W(x,1)')
    # plt.title('Comparison of Whittle indices (discounted)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(8,5))
    # plt.plot(range(N+1), W_closed_avg, 'o-', label='Closed-form (avg)')
    # plt.plot(range(N+1), ind_avg_rvi_active, 's--', label='RVI+bisect (avg)')
    # if pkg_avg_active is not None:
    #     plt.plot(range(N+1), pkg_avg_active, '^:', label='markovianbandit (avg)')
    # plt.xlabel('Deterioration state x')
    # plt.ylabel('Whittle index W(x,1)')
    # plt.title('Comparison of Whittle indices (average cost)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    #plot_transitions(P0, P1)

if __name__ == "__main__":
    main()