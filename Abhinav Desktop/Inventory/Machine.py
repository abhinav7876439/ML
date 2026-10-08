"""
Machine Repairman Problem with Switching Costs
— Restless Bandit Formulation & Whittle Index Computation —

Instances:
  (A) M=3 machines, R=1 repairman
  (B) M=4 machines, R=2 repairmen
Each machine: 8 physical states  {0,1,...,7}  (N=7)
Extended state per machine: (x, a_prev) ∈ {0,...,7} × {0,1}  → 16 states
"""

import numpy as np


# ──────────────────────────────────────────────────────────
#  Single-Machine Solver (parameterised by subsidy w)
# ──────────────────────────────────────────────────────────

class SingleMachineRB:
    """
    One machine modelled as a Restless Bandit arm with switching cost.

    Extended state  l = (x, a_prev)
        x      ∈ {0, 1, ..., N}   deterioration level
        a_prev ∈ {0, 1}           previous-epoch action (0=passive, 1=active)

    Actions:  1 = active (repair),  0 = passive (leave unattended)
    """

    def __init__(self, N, Cm, Km, c_sw, p, beta=0.99):
        self.N     = N
        self.Cm    = Cm
        self.Km    = Km
        self.c_sw  = c_sw
        self.p     = np.asarray(p, dtype=float)
        self.beta  = beta

        self.states = [(x, ap) for x in range(N + 1) for ap in (0, 1)]
        self.n      = len(self.states)
        self._idx   = {s: i for i, s in enumerate(self.states)}

    def _si(self, x, ap):
        return self._idx[(x, ap)]

    def _build(self, w):
        """Build cost vectors and transition matrices for both actions."""
        n = self.n
        c_a = np.zeros(n)
        c_b = np.zeros(n)
        P_a = np.zeros((n, n))
        P_b = np.zeros((n, n))

        for x in range(self.N + 1):
            for ap in (0, 1):
                s = self._si(x, ap)

                # ---- active action ----
                c_a[s] = self.Cm + w + (self.c_sw if ap == 0 else 0.0)
                P_a[s, self._si(0, 1)] = 1.0

                # ---- passive action ----
                px = self.p[x] if x < self.N else 0.0   # Survival probability (increasing order) at deterioration level x
                c_b[s] = self.Km * (1.0 - px)   # (1 - px) probability of breakdown 
                if x < self.N:
                    P_b[s, self._si(x + 1, 0)] += px
                    P_b[s, self._si(0, 0)]     += 1.0 - px
                else:
                    # at max state, guaranteed breakdown
                    P_b[s, self._si(0, 0)] = 1.0

        return c_a, c_b, P_a, P_b


    # ---------- discounted-cost value iteration ----------
    def vi_discounted(self, w, tol=1e-10, maxiter=100_000):
        c_a, c_b, P_a, P_b = self._build(w)
        V = np.zeros(self.n)
        for _ in range(maxiter):
            Qa = c_a + self.beta * (P_a @ V)
            Qb = c_b + self.beta * (P_b @ V)
            Vnew = np.minimum(Qa, Qb)
            if np.max(np.abs(Vnew - V)) < tol:
                V = Vnew
                break
            V = Vnew
        Qa = c_a + self.beta * (P_a @ V)
        Qb = c_b + self.beta * (P_b @ V)
        policy = (Qa <= Qb).astype(int)
        return V, policy

    # ---------- average-cost via span-based RVI ----------
    def rvi_average(self, w, tol=1e-10, maxiter=500_000):
        """
        Relative Value Iteration with SPAN SEMINORM convergence check.
        
        Uses h(s) - h(ref) normalization at every step.
        Convergence: span(h_new - h) = max(h_new-h) - min(h_new-h) < tol
        """
        c_a, c_b, P_a, P_b = self._build(w)
        h = np.zeros(self.n)
        ref = 0  # reference state index
        g = 0.0

        for iteration in range(maxiter):
            Qa = c_a + P_a @ h
            Qb = c_b + P_b @ h
            h_new = np.minimum(Qa, Qb)

            # Extract average cost as shift at reference state
            g_new = h_new[ref]
            h_new = h_new - g_new  # normalize

            # Span seminorm convergence check
            diff = h_new - h
            span = np.max(diff) - np.min(diff)

            if span < tol:
                h = h_new
                g = g_new
                break
            h = h_new
            g = g_new

        # Final policy
        Qa = c_a + P_a @ h
        Qb = c_b + P_b @ h
        policy = (Qa <= Qb).astype(int)
        return h, policy, g

    # ---------- average-cost via discounted approximation ----------
    def avg_cost_via_discount(self, w, beta_high=0.9999, tol=1e-10, maxiter=200_000):
        """
        Approximate average cost using:  g ≈ (1 - β) * V_β  with β close to 1.
        More robust than RVI for some problem structures.
        """
        old_beta = self.beta
        self.beta = beta_high
        V, policy = self.vi_discounted(w, tol=tol, maxiter=maxiter)
        g_approx = (1.0 - beta_high) * V
        self.beta = old_beta
        return V, policy, g_approx


# ──────────────────────────────────────────────────────────
#  Whittle Index Computation (Binary Search)
# ──────────────────────────────────────────────────────────

def whittle_indices_discounted(arm: SingleMachineRB,
                               wlo=-300, whi=300, tol=1e-6):
    """Return dict {(x, a_prev): W_index} under β-discounted cost."""
    indices = {}
    _, pol_lo = arm.vi_discounted(wlo)
    _, pol_hi = arm.vi_discounted(whi)

    for si in range(arm.n):
        state = arm.states[si]

        if pol_lo[si] == 0:
            indices[state] = wlo
            continue
        if pol_hi[si] == 1:
            indices[state] = whi
            continue

        lo, hi = wlo, whi
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            _, pol = arm.vi_discounted(mid)
            if pol[si] == 1:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        indices[state] = 0.5 * (lo + hi)
    return indices


def whittle_indices_average(arm: SingleMachineRB,
                            wlo=-300, whi=300, tol=1e-6,
                            method='rvi'):
    """
    Return dict {(x, a_prev): W_index} under average cost.
    method: 'rvi' = relative value iteration, 'discount' = discounted approx
    """
    indices = {}

    def get_policy(w):
        if method == 'rvi':
            _, pol, _ = arm.rvi_average(w)
        else:
            _, pol, _ = arm.avg_cost_via_discount(w)
        return pol

    pol_lo = get_policy(wlo)
    pol_hi = get_policy(whi)

    for si in range(arm.n):
        state = arm.states[si]

        if pol_lo[si] == 0:
            indices[state] = wlo
            continue
        if pol_hi[si] == 1:
            indices[state] = whi
            continue

        lo, hi = wlo, whi
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            pol = get_policy(mid)
            if pol[si] == 1:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        indices[state] = 0.5 * (lo + hi)
    return indices


# ──────────────────────────────────────────────────────────
#  Multi-Machine Simulation under Whittle-Index Policy
# ──────────────────────────────────────────────────────────

def simulate_whittle_policy(arms, R, W_tables, T=100_000,
                            beta=0.99, seed=42):
    """
    Simulate M machines / R repairmen using precomputed Whittle indices.
    """
    rng = np.random.default_rng(seed)
    M   = len(arms)

    xs      = np.zeros(M, dtype=int)
    a_prevs = np.zeros(M, dtype=int)

    total_disc_cost = 0.0
    total_cost      = 0.0

    for t in range(T):
        # rank by Whittle index descending
        w_vals = np.array([W_tables[m][(xs[m], a_prevs[m])] for m in range(M)])
        ranked = np.argsort(-w_vals)

        actions = np.zeros(M, dtype=int)
        for j in range(min(R, M)):
            actions[ranked[j]] = 1

        # costs
        period_cost = 0.0
        for m in range(M):
            arm_m = arms[m]
            x, ap = xs[m], a_prevs[m]
            if actions[m] == 1:
                c = arm_m.Cm + (arm_m.c_sw if ap == 0 else 0.0)
            else:
                px = arm_m.p[x] if x < arm_m.N else 0.0
                c = arm_m.Km * (1.0 - px)
            period_cost += c

        total_disc_cost += (beta ** t) * period_cost
        total_cost      += period_cost

        # transitions
        new_xs  = np.zeros(M, dtype=int)
        new_aps = actions.copy()
        for m in range(M):
            arm_m = arms[m]
            x = xs[m]
            if actions[m] == 1:
                new_xs[m] = 0
            else:
                px = arm_m.p[x] if x < arm_m.N else 0.0
                if rng.random() < px:
                    new_xs[m] = x + 1
                else:
                    new_xs[m] = 0
        xs      = new_xs
        a_prevs = new_aps

    avg_cost = total_cost / T
    return total_disc_cost, avg_cost


# ──────────────────────────────────────────────────────────
#  Pretty-Print Helpers
# ──────────────────────────────────────────────────────────

def print_state_space(N):
    print(f"\n{'='*65}")
    print(f"  EXTENDED STATE SPACE  (N = {N}, physical states 0..{N})")
    print(f"{'='*65}")
    print(f"  Total extended states per machine: {2*(N+1)}")
    print(f"\n  Active-family  (a_prev=1, repairman was here last period):")
    print(f"    ", [(x, 1) for x in range(N+1)])
    print(f"  Passive-family (a_prev=0, repairman was NOT here last period):")
    print(f"    ", [(x, 0) for x in range(N+1)])
    print()


def print_indices(title, idx_dict, N):
    print(f"\n  {title}")
    print(f"  {'State (x, a_prev)':>22s}  {'Whittle Index':>14s}")
    print(f"  {'-'*22}  {'-'*14}")
    for x in range(N + 1):
        for ap in (0, 1):
            w = idx_dict.get((x, ap), None)
            fam = 'passive' if ap == 0 else 'active'
            if w is not None:
                print(f"  {f'({x}, {ap})  [{fam:>7s}]':>22s}  {w:>14.4f}")
    print()


def check_indexability(arm, wlo=-200, whi=200, n_test=300):
    """Verify that as w increases the active set shrinks monotonically."""
    ws = np.linspace(wlo, whi, n_test)
    active_sets = []
    for w in ws:
        _, pol = arm.vi_discounted(w)
        active_sets.append(set(np.where(pol == 1)[0]))

    indexable = True
    for i in range(1, len(ws)):
        if not active_sets[i].issubset(active_sets[i - 1]):
            indexable = False
            print(f"  ✗ Indexability VIOLATED between w={ws[i-1]:.2f} "
                  f"and w={ws[i]:.2f}")
            break
    if indexable:
        print(f"  ✓ Indexability VERIFIED (active set shrinks monotonically "
              f"over {n_test} test values of w in [{wlo}, {whi}])")


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    # ============ PARAMETERS ============
    N    = 7       # states 0..7 (8 physical states)
    beta = 0.95

    # Survival probabilities (decreasing)
    p = np.array([0.95, 0.90, 0.85, 0.78, 0.70, 0.55, 0.35, 0.0])

    # Costs
    Cm   = 5.0      # preventive repair cost
    Km   = 100.0    # catastrophic breakdown cost  (Km >> Cm)
    c_sw = 3.0      # switching cost

    # ============ PRINT STATE SPACE ============
    print_state_space(N)

    # ============ BUILD ARM ============
    arm = SingleMachineRB(N=N, Cm=Cm, Km=Km, c_sw=c_sw, p=p, beta=beta)

    Active_cost, Passive_cost, Prob_active, Prob_passive = arm._build(w=0.0)
    print("active cost vector (c_a):", Active_cost)
    print("passive cost vector (c_b):", Passive_cost)
    print("transition matrix for active action (P_a):")
    print(Prob_active)
    print("transition matrix for passive action (P_b):")
    print(Prob_passive)

    # ============ INDEXABILITY CHECK ============
    print("=" * 65)
    print("  INDEXABILITY CHECK (discounted)")
    print("=" * 65)
    check_indexability(arm)

    # ============ DISCOUNTED-COST WHITTLE INDICES ============
    print("\n" + "=" * 65)
    print(f"  DISCOUNTED-COST WHITTLE INDICES  (β = {beta})")
    print("=" * 65)
    W_disc = whittle_indices_discounted(arm)
    print_indices("Discounted Whittle Indices", W_disc, N)

    # ============ AVERAGE-COST WHITTLE INDICES ============
    # Using discounted approximation (robust, no convergence issues)
    print("=" * 65)
    print("  AVERAGE-COST WHITTLE INDICES  (via RVI with span seminorm)")
    print("=" * 65)

    try:
        W_avg = whittle_indices_average(arm, method='rvi')
        print_indices("Average-Cost Whittle Indices (RVI)", W_avg, N)
        rvi_ok = True
    except Exception as e:
        print(f"  RVI failed: {e}")
        rvi_ok = False

    # Fallback / comparison: discounted approximation β → 1
    print("=" * 65)
    print("  AVERAGE-COST WHITTLE INDICES  (via discounted approx, β=0.9999)")
    print("=" * 65)
    W_avg_disc = whittle_indices_average(arm, method='discount')
    print_indices("Average-Cost Whittle Indices (Discount Approx)", W_avg_disc, N)

    # Use whichever average-cost indices succeeded
    W_avg_use = W_avg if rvi_ok else W_avg_disc

    # ============ INSTANCE A : 3 machines, 1 repairman ============
    M_A, R_A = 3, 1
    arms_A          = [arm] * M_A
    W_disc_tables_A = [W_disc] * M_A
    W_avg_tables_A  = [W_avg_use] * M_A

    print("\n" + "=" * 65)
    print(f"  INSTANCE A :  M = {M_A} machines,  R = {R_A} repairman")
    print("=" * 65)

    disc_cost_A, avg_cost_A_d = simulate_whittle_policy(
        arms_A, R_A, W_disc_tables_A, T=200_000, beta=beta)
    print(f"  Whittle-index policy (DISCOUNTED indices):")
    print(f"    Total discounted cost (T=200k):  {disc_cost_A:>12.2f}")
    print(f"    Average cost per period:          {avg_cost_A_d:>12.4f}")

    disc_cost_A2, avg_cost_A_a = simulate_whittle_policy(
        arms_A, R_A, W_avg_tables_A, T=200_000, beta=beta)
    print(f"  Whittle-index policy (AVERAGE-COST indices):")
    print(f"    Total discounted cost (T=200k):  {disc_cost_A2:>12.2f}")
    print(f"    Average cost per period:          {avg_cost_A_a:>12.4f}")

    # ============ INSTANCE B : 4 machines, 2 repairmen ============
    M_B, R_B = 4, 2
    arms_B          = [arm] * M_B
    W_disc_tables_B = [W_disc] * M_B
    W_avg_tables_B  = [W_avg_use] * M_B

    print("\n" + "=" * 65)
    print(f"  INSTANCE B :  M = {M_B} machines,  R = {R_B} repairmen")
    print("=" * 65)

    disc_cost_B, avg_cost_B_d = simulate_whittle_policy(
        arms_B, R_B, W_disc_tables_B, T=200_000, beta=beta)
    print(f"  Whittle-index policy (DISCOUNTED indices):")
    print(f"    Total discounted cost (T=200k):  {disc_cost_B:>12.2f}")
    print(f"    Average cost per period:          {avg_cost_B_d:>12.4f}")

    disc_cost_B2, avg_cost_B_a = simulate_whittle_policy(
        arms_B, R_B, W_avg_tables_B, T=200_000, beta=beta)
    print(f"  Whittle-index policy (AVERAGE-COST indices):")
    print(f"    Total discounted cost (T=200k):  {disc_cost_B2:>12.2f}")
    print(f"    Average cost per period:          {avg_cost_B_a:>12.4f}")

    # ============ SUMMARY TABLE ============
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'':30s} {'Instance A':>15s} {'Instance B':>15s}")
    print(f"  {'':30s} {'(M=3,R=1)':>15s} {'(M=4,R=2)':>15s}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Disc. policy: disc. cost':30s} {disc_cost_A:>15.2f} {disc_cost_B:>15.2f}")
    print(f"  {'Disc. policy: avg cost/period':30s} {avg_cost_A_d:>15.4f} {avg_cost_B_d:>15.4f}")
    print(f"  {'Avg. policy: avg cost/period':30s} {avg_cost_A_a:>15.4f} {avg_cost_B_a:>15.4f}")
    print()


if __name__ == "__main__":
    main()