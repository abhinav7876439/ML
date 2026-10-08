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
from itertools import combinations

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

    Costs (with intervention charge w):
        Active from (x,1):  Cm + w
        Active from (x,0):  Cm + w + c_sw
        Passive from (x,*): Km * (1 - p[x])   (expected breakdown cost)

    Transitions:
        Active  → (0, 1)  deterministically
        Passive → (x+1, 0) w.p. p[x],  (0, 0) w.p. 1-p[x]
                  (at x=N: guaranteed breakdown → (0,0))
    """

    def __init__(self, N, Cm, Km, c_sw, p, beta=0.99):
        self.N     = N
        self.Cm    = Cm
        self.Km    = Km
        self.c_sw  = c_sw
        self.p     = np.asarray(p, dtype=float)   # length N+1
        self.beta  = beta

        # Build state list and index map
        self.states = [(x, ap) for x in range(N+1) for ap in (0, 1)]
        self.n      = len(self.states)             # 2*(N+1)
        self._idx   = {s: i for i, s in enumerate(self.states)}

    def _si(self, x, ap):
        return self._idx[(x, ap)]

    # ---------- build cost & transition structures ----------
    def _build(self, w):
        n = self.n
        c_a = np.zeros(n)
        c_b = np.zeros(n)
        P_a = np.zeros((n, n))
        P_b = np.zeros((n, n))

        for x in range(self.N + 1):
            for ap in (0, 1):
                s = self._si(x, ap)

                # ---- active ----
                c_a[s] = self.Cm + w + (self.c_sw if ap == 0 else 0.0)
                P_a[s, self._si(0, 1)] = 1.0

                # ---- passive ----
                px = self.p[x] if x < self.N else 0.0
                c_b[s] = self.Km * (1.0 - px)
                if x < self.N:
                    P_b[s, self._si(x + 1, 0)] += px
                    P_b[s, self._si(0, 0)]     += 1.0 - px
                else:
                    P_b[s, self._si(0, 0)] = 1.0

        return c_a, c_b, P_a, P_b

    # ---------- discounted-cost value iteration ----------
    def vi_discounted(self, w, tol=1e-12, maxiter=200_000):
        c_a, c_b, P_a, P_b = self._build(w)
        V = np.zeros(self.n)
        for _ in range(maxiter):
            Qa = c_a + self.beta * (P_a @ V)
            Qb = c_b + self.beta * (P_b @ V)
            Vnew = np.minimum(Qa, Qb)
            if np.max(np.abs(Vnew - V)) < tol:
                break
            V = Vnew
        Qa = c_a + self.beta * (P_a @ V)
        Qb = c_b + self.beta * (P_b @ V)
        policy = (Qa <= Qb).astype(int)        # 1=active, 0=passive
        return V, policy

    # ---------- average-cost relative value iteration ----------
    def rvi_average(self, w, tol=1e-12, maxiter=300_000):
        c_a, c_b, P_a, P_b = self._build(w)
        h = np.zeros(self.n)
        ref = 0
        for _ in range(maxiter):
            Qa = c_a + P_a @ h
            Qb = c_b + P_b @ h
            h_new = np.minimum(Qa, Qb)
            g = h_new[ref]              # average cost estimate
            h_new -= g                  # normalise
            if np.max(np.abs(h_new - h)) < tol:
                break
            h = h_new
        Qa = c_a + P_a @ h
        Qb = c_b + P_b @ h
        policy = (Qa <= Qb).astype(int)
        return h, policy, g


# ──────────────────────────────────────────────────────────
#  Whittle Index Computation (Binary Search)
# ──────────────────────────────────────────────────────────

def whittle_indices_discounted(arm: SingleMachineRB,
                               wlo=-500, whi=500, tol=1e-6):
    """Return dict  {(x,a_prev): W_index} under β-discounted cost."""
    indices = {}
    # precompute boundary policies
    _, pol_lo = arm.vi_discounted(wlo)
    _, pol_hi = arm.vi_discounted(whi)

    for si in range(arm.n):
        x, ap = arm.states[si]
        if pol_lo[si] == 0:          # passive even at lowest w
            indices[(x, ap)] = wlo
            continue
        if pol_hi[si] == 1:          # active even at highest w
            indices[(x, ap)] = whi
            continue
        lo, hi = wlo, whi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            _, pol = arm.vi_discounted(mid)
            if pol[si] == 1:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        indices[(x, ap)] = 0.5 * (lo + hi)
    return indices


def whittle_indices_average(arm: SingleMachineRB,
                            wlo=-500, whi=500, tol=1e-6):
    """Return dict  {(x,a_prev): W_index} under average cost."""
    indices = {}
    _, pol_lo, _ = arm.rvi_average(wlo)
    _, pol_hi, _ = arm.rvi_average(whi)

    for si in range(arm.n):
        x, ap = arm.states[si]
        if pol_lo[si] == 0:
            indices[(x, ap)] = wlo
            continue
        if pol_hi[si] == 1:
            indices[(x, ap)] = whi
            continue
        lo, hi = wlo, whi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            _, pol, _ = arm.rvi_average(mid)
            if pol[si] == 1:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        indices[(x, ap)] = 0.5 * (lo + hi)
    return indices


# ──────────────────────────────────────────────────────────
#  Multi-Machine Simulation under Whittle-Index Policy
# ──────────────────────────────────────────────────────────

def simulate_whittle_policy(arms, R, W_tables, T=100_000,
                            beta=0.99, seed=42, discounted=True):
    """
    Simulate M machines / R repairmen using precomputed Whittle indices.

    arms     : list of SingleMachineRB (length M)
    R        : number of repairmen
    W_tables : list of dicts {(x,ap): index} for each arm
    T        : horizon
    Returns total discounted cost (or average cost per period).
    """
    rng = np.random.default_rng(seed)
    M   = len(arms)

    # initial extended state: all pristine, no repairman assigned
    xs    = np.zeros(M, dtype=int)
    a_prevs = np.zeros(M, dtype=int)

    total_disc_cost = 0.0
    total_cost      = 0.0

    for t in range(T):
        # ---- rank machines by Whittle index (descending) ----
        w_vals = np.array([W_tables[m][(xs[m], a_prevs[m])] for m in range(M)])
        ranked = np.argsort(-w_vals)          # highest index first

        actions = np.zeros(M, dtype=int)      # 0=passive
        for j in range(min(R, M)):
            actions[ranked[j]] = 1             # assign repairman

        # ---- costs ----
        period_cost = 0.0
        for m in range(M):
            arm = arms[m]
            x, ap = xs[m], a_prevs[m]
            if actions[m] == 1:
                c = arm.Cm + (arm.c_sw if ap == 0 else 0.0)
            else:
                px = arm.p[x] if x < arm.N else 0.0
                c = arm.Km * (1.0 - px)
            period_cost += c

        total_disc_cost += (beta ** t) * period_cost
        total_cost      += period_cost

        # ---- transitions ----
        new_xs    = np.zeros(M, dtype=int)
        new_aps   = actions.copy()
        for m in range(M):
            arm = arms[m]
            x   = xs[m]
            if actions[m] == 1:
                new_xs[m] = 0
            else:
                px = arm.p[x] if x < arm.N else 0.0
                if rng.random() < px:
                    new_xs[m] = x + 1
                else:
                    new_xs[m] = 0     # catastrophic breakdown
        xs      = new_xs
        a_prevs = new_aps

    avg_cost = total_cost / T
    return total_disc_cost, avg_cost


# ──────────────────────────────────────────────────────────
#  Pretty-Print Helpers
# ──────────────────────────────────────────────────────────

def print_state_space(N):
    print(f"\n{'='*60}")
    print(f"  EXTENDED STATE SPACE  (N = {N}, physical states 0..{N})")
    print(f"{'='*60}")
    print(f"  Total extended states per machine: {2*(N+1)}")
    print(f"\n  Active-family  (a_prev=1, repairman was here):")
    print(f"    ", [(x, 1) for x in range(N+1)])
    print(f"  Passive-family (a_prev=0, repairman was NOT here):")
    print(f"    ", [(x, 0) for x in range(N+1)])
    print()


def print_indices(title, idx_dict, N):
    print(f"\n  {title}")
    print(f"  {'State (x,a_prev)':>20s}  {'Whittle Index':>14s}")
    print(f"  {'-'*20}  {'-'*14}")
    for x in range(N+1):
        for ap in (0, 1):
            w = idx_dict.get((x, ap), None)
            label = f"({'passive' if ap==0 else 'active':>7s})"
            if w is not None:
                print(f"  {f'({x}, {ap}) {label}':>20s}  {w:>14.4f}")


# ──────────────────────────────────────────────────────────
#  MAIN — Run both instances
# ──────────────────────────────────────────────────────────

def main():
    # ============ PARAMETERS ============
    N     = 7                     # states 0..7  (8 states)
    beta  = 0.95                  # discount factor

    # Survival probabilities (decreasing → higher states more dangerous)
    p = np.array([0.95, 0.90, 0.85, 0.78, 0.70, 0.55, 0.35, 0.0])
    #             p[0]  p[1]  p[2]  p[3]  p[4]  p[5]  p[6]  p[7]

    # Cost parameters (same for every machine here; easily made heterogeneous)
    Cm    = 5.0       # preventive repair cost
    Km    = 100.0     # catastrophic breakdown cost  (Km >> Cm)
    c_sw  = 3.0       # switching cost

    # ============ PRINT STATE SPACE ============
    print_state_space(N)

    # ============ BUILD SINGLE-MACHINE ARM ============
    arm = SingleMachineRB(N=N, Cm=Cm, Km=Km, c_sw=c_sw, p=p, beta=beta)

    # ============ DISCOUNTED-COST WHITTLE INDICES ============
    print("=" * 60)
    print("  DISCOUNTED-COST WHITTLE INDICES  (β = {:.2f})".format(beta))
    print("=" * 60)
    W_disc = whittle_indices_discounted(arm)
    print_indices("Discounted Whittle Indices", W_disc, N)

    # ============ AVERAGE-COST WHITTLE INDICES ============
    print("\n" + "=" * 60)
    print("  AVERAGE-COST WHITTLE INDICES")
    print("=" * 60)
    W_avg = whittle_indices_average(arm)
    print_indices("Average-Cost Whittle Indices", W_avg, N)

    # ============ INSTANCE A : 3 machines, 1 repairman ============
    M_A, R_A = 3, 1
    arms_A   = [arm] * M_A
    W_disc_tables_A = [W_disc] * M_A
    W_avg_tables_A  = [W_avg]  * M_A

    disc_cost_A, avg_cost_A = simulate_whittle_policy(
        arms_A, R_A, W_disc_tables_A, T=200_000, beta=beta, discounted=True)
    _, avg_cost_A2 = simulate_whittle_policy(
        arms_A, R_A, W_avg_tables_A, T=200_000, beta=beta, discounted=False)

    print("\n" + "=" * 60)
    print(f"  INSTANCE A :  M={M_A} machines,  R={R_A} repairman")
    print("=" * 60)
    print(f"  Whittle-index policy (discounted indices):")
    print(f"    Total discounted cost (T=200k): {disc_cost_A:>14.2f}")
    print(f"    Average cost per period:        {avg_cost_A:>14.4f}")
    print(f"  Whittle-index policy (average-cost indices):")
    print(f"    Average cost per period:        {avg_cost_A2:>14.4f}")

    # ============ INSTANCE B : 4 machines, 2 repairmen ============
    M_B, R_B = 4, 2
    arms_B   = [arm] * M_B
    W_disc_tables_B = [W_disc] * M_B
    W_avg_tables_B  = [W_avg]  * M_B

    disc_cost_B, avg_cost_B = simulate_whittle_policy(
        arms_B, R_B, W_disc_tables_B, T=200_000, beta=beta, discounted=True)
    _, avg_cost_B2 = simulate_whittle_policy(
        arms_B, R_B, W_avg_tables_B, T=200_000, beta=beta, discounted=False)

    print("\n" + "=" * 60)
    print(f"  INSTANCE B :  M={M_B} machines,  R={R_B} repairmen")
    print("=" * 60)
    print(f"  Whittle-index policy (discounted indices):")
    print(f"    Total discounted cost (T=200k): {disc_cost_B:>14.2f}")
    print(f"    Average cost per period:        {avg_cost_B:>14.4f}")
    print(f"  Whittle-index policy (average-cost indices):")
    print(f"    Average cost per period:        {avg_cost_B2:>14.4f}")

    # ============ VERIFY INDEXABILITY ============
    print("\n" + "=" * 60)
    print("  INDEXABILITY CHECK")
    print("=" * 60)
    check_indexability(arm)


def check_indexability(arm, wlo=-200, whi=200, n_test=300):
    """
    Verify indexability: as w increases the active set should shrink monotonically.
    """
    ws = np.linspace(wlo, whi, n_test)
    active_sets = []
    for w in ws:
        _, pol = arm.vi_discounted(w)
        active_sets.append(set(np.where(pol == 1)[0]))

    indexable = True
    for i in range(1, len(ws)):
        if not active_sets[i].issubset(active_sets[i-1]):
            indexable = False
            print(f"  ✗ Indexability VIOLATED between w={ws[i-1]:.2f} "
                  f"and w={ws[i]:.2f}")
            break
    if indexable:
        print("  ✓ Indexability VERIFIED (active set shrinks monotonically "
              f"over {n_test} test values of w)")


if __name__ == "__main__":
    main()