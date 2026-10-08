"""
Machine Repairman Problem with Switching Costs
— Restless Bandit Formulation & Whittle Index Computation —

Instances:
  (A) M=3 machines, R=1 repairman
  (B) M=4 machines, R=2 repairmen
Each machine: 8 physical states {0,1,...,7} (N=7)
Extended state: (x, a_prev) ∈ {0,...,7} × {0,1} → 16 states
"""

import numpy as np
import time


class SingleMachineRB:
    def __init__(self, N, Cm, Km, c_sw, p, beta=0.95):
        self.N     = N
        self.Cm    = Cm
        self.Km    = Km
        self.c_sw  = c_sw
        self.p     = np.asarray(p, dtype=float)
        self.beta  = beta

        # Build state list: (x, a_prev) for x in 0..N, a_prev in {0,1}
        self.states = []
        for x in range(N + 1):
            for ap in (0, 1):
                self.states.append((x, ap))
        self.n = len(self.states)          # = 2*(N+1) = 16
        self._idx = {s: i for i, s in enumerate(self.states)}

        # ─── PRECOMPUTE everything that does NOT depend on w ───
        self._precompute()

    def _si(self, x, ap):
        """Map (x, a_prev) → integer index."""
        return self._idx[(x, ap)]

    def _precompute(self):
        """
        Precompute transition matrices P_a, P_b and passive cost c_b.
        These are INDEPENDENT of the subsidy w.
        Only active cost c_a depends on w (recomputed per call).
        """
        n = self.n

        # ── Transition matrices ──
        self.P_a = np.zeros((n, n))
        self.P_b = np.zeros((n, n))
        self.c_b = np.zeros(n)

        # Also precompute the w-independent part of c_a
        self.c_a_base = np.zeros(n)  # Cm + (c_sw if ap==0 else 0)

        for x in range(self.N + 1):
            for ap in (0, 1):
                s = self._si(x, ap)

                # ── Active action transitions ──
                # Always go to (0, 1) with probability 1
                self.P_a[s, self._si(0, 1)] = 1.0

                # Active cost (without w): Cm + switching cost if needed
                self.c_a_base[s] = self.Cm + (self.c_sw if ap == 0 else 0.0)

                # ── Passive action transitions ──
                px = self.p[x] if x < self.N else 0.0

                # Passive cost: expected breakdown cost
                self.c_b[s] = self.Km * (1.0 - px)

                if x < self.N:
                    # Survive: x → x+1, a_prev becomes 0
                    self.P_b[s, self._si(x + 1, 0)] = px
                    # Breakdown: → (0, 0)
                    self.P_b[s, self._si(0, 0)] = 1.0 - px
                else:
                    # At max state: guaranteed breakdown → (0, 0)
                    self.P_b[s, self._si(0, 0)] = 1.0

        print(f"  Precomputed: n={n} states, P_a, P_b, c_b, c_a_base")
        print(f"  P_a row sums: {self.P_a.sum(axis=1)}")
        print(f"  P_b row sums: {self.P_b.sum(axis=1)}")

    def _get_c_a(self, w):
        """Active cost = base + w. Only this changes with w."""
        return self.c_a_base + w

    # ────────── Discounted-Cost Value Iteration ──────────
    def vi_discounted(self, w, tol=1e-8, maxiter=5000):
        c_a = self._get_c_a(w)
        c_b = self.c_b
        P_a = self.P_a
        P_b = self.P_b
        beta = self.beta

        V = np.zeros(self.n)

        for it in range(maxiter):
            Qa = c_a + beta * (P_a @ V)
            Qb = c_b + beta * (P_b @ V)
            Vnew = np.minimum(Qa, Qb)

            diff = np.max(np.abs(Vnew - V))
            if diff < tol:
                V = Vnew
                break
            V = Vnew

        # Compute final policy
        Qa = c_a + beta * (P_a @ V)
        Qb = c_b + beta * (P_b @ V)
        policy = (Qa <= Qb).astype(int)   # 1 = active, 0 = passive
        return V, policy, it + 1

    # ────────── Average-Cost Relative Value Iteration ──────────
    def rvi_average(self, w, tol=1e-8, maxiter=50000):
        c_a = self._get_c_a(w)
        c_b = self.c_b
        P_a = self.P_a
        P_b = self.P_b

        h = np.zeros(self.n)
        ref = 0
        g = 0.0

        for it in range(maxiter):
            Qa = c_a + P_a @ h
            Qb = c_b + P_b @ h
            h_new = np.minimum(Qa, Qb)

            g = h_new[ref]
            h_new = h_new - g       # normalize by reference state

            # Span seminorm convergence
            diff = h_new - h
            span = np.max(diff) - np.min(diff)

            if span < tol:
                h = h_new
                break
            h = h_new

        # Final policy
        Qa = c_a + P_a @ h
        Qb = c_b + P_b @ h
        policy = (Qa <= Qb).astype(int)
        return h, policy, g, it + 1


# ──────────────────────────────────────────────────────────
#  Whittle Index Computation
# ──────────────────────────────────────────────────────────

def whittle_indices_discounted(arm, wlo=-200.0, whi=200.0, tol=1e-5):
    """Binary search for Whittle index of each extended state (discounted)."""
    indices = {}

    _, pol_lo, _ = arm.vi_discounted(wlo)
    _, pol_hi, _ = arm.vi_discounted(whi)

    for si in range(arm.n):
        state = arm.states[si]

        # If passive even at lowest w → index = wlo (boundary)
        if pol_lo[si] == 0:
            indices[state] = wlo
            continue
        # If active even at highest w → index = whi (boundary)
        if pol_hi[si] == 1:
            indices[state] = whi
            continue

        # Binary search
        lo, hi = wlo, whi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            _, pol, _ = arm.vi_discounted(mid)
            if pol[si] == 1:   # active at mid → need higher w to make passive
                lo = mid
            else:              # passive at mid → need lower w
                hi = mid
            if (hi - lo) < tol:
                break
        indices[state] = 0.5 * (lo + hi)

    return indices


def whittle_indices_average(arm, wlo=-200.0, whi=200.0, tol=1e-5):
    """Binary search for Whittle index of each extended state (average cost)."""
    indices = {}

    _, pol_lo, _, _ = arm.rvi_average(wlo)
    _, pol_hi, _, _ = arm.rvi_average(whi)

    for si in range(arm.n):
        state = arm.states[si]

        if pol_lo[si] == 0:
            indices[state] = wlo
            continue
        if pol_hi[si] == 1:
            indices[state] = whi
            continue

        lo, hi = wlo, whi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            _, pol, _, _ = arm.rvi_average(mid)
            if pol[si] == 1:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol:
                break
        indices[state] = 0.5 * (lo + hi)

    return indices


# ──────────────────────────────────────────────────────────
#  Multi-Machine Simulation
# ──────────────────────────────────────────────────────────

def simulate_whittle_policy(arms, R, W_tables, T=100000, beta=0.95, seed=42):
    rng = np.random.default_rng(seed)
    M   = len(arms)

    xs      = np.zeros(M, dtype=int)      # physical states
    a_prevs = np.zeros(M, dtype=int)      # previous actions

    total_disc_cost = 0.0
    total_cost      = 0.0

    for t in range(T):
        # Rank by Whittle index (descending)
        w_vals = np.array([W_tables[m][(xs[m], a_prevs[m])] for m in range(M)])
        ranked = np.argsort(-w_vals)

        actions = np.zeros(M, dtype=int)
        for j in range(min(R, M)):
            actions[ranked[j]] = 1

        # Compute period cost
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

        # State transitions
        new_xs  = np.zeros(M, dtype=int)
        new_aps = actions.copy()
        for m in range(M):
            arm_m = arms[m]
            x = xs[m]
            if actions[m] == 1:
                new_xs[m] = 0       # repair → pristine
            else:
                px = arm_m.p[x] if x < arm_m.N else 0.0
                if rng.random() < px:
                    new_xs[m] = x + 1   # survive, deteriorate
                else:
                    new_xs[m] = 0       # breakdown, reset

        xs      = new_xs
        a_prevs = new_aps

    return total_disc_cost, total_cost / T


# ──────────────────────────────────────────────────────────
#  Display Helpers
# ──────────────────────────────────────────────────────────

def print_state_space(N):
    print(f"\n{'='*65}")
    print(f"  EXTENDED STATE SPACE  (N={N}, physical states 0..{N})")
    print(f"{'='*65}")
    print(f"  Total extended states per machine: {2*(N+1)}")
    print(f"\n  Passive family (a_prev=0, repairman was NOT here):")
    for x in range(N + 1):
        print(f"    ({x}, 0)", end="")
    print()
    print(f"  Active family  (a_prev=1, repairman WAS here):")
    for x in range(N + 1):
        print(f"    ({x}, 1)", end="")
    print("\n")


def print_indices(title, idx_dict, N):
    print(f"\n  {title}")
    print(f"  {'State':>12s} {'Family':>10s} {'Whittle Index':>15s}")
    print(f"  {'-'*12} {'-'*10} {'-'*15}")
    for x in range(N + 1):
        for ap in (0, 1):
            w = idx_dict.get((x, ap), float('nan'))
            fam = 'active' if ap == 1 else 'passive'
            print(f"  {f'({x}, {ap})':>12s} {fam:>10s} {w:>15.6f}")
    print()


def check_indexability(arm, wlo=-200, whi=200, n_test=200):
    """Verify that active set shrinks monotonically as w increases."""
    ws = np.linspace(wlo, whi, n_test)
    prev_active = None
    ok = True
    for w in ws:
        _, pol, _ = arm.vi_discounted(w)
        curr_active = set(np.where(pol == 1)[0])
        if prev_active is not None:
            if not curr_active.issubset(prev_active):
                print(f"  ✗ Indexability VIOLATED at w={w:.2f}")
                ok = False
                break
        prev_active = curr_active
    if ok:
        print(f"  ✓ Indexability VERIFIED ({n_test} test points in [{wlo},{whi}])")
    return ok


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    # ============ PARAMETERS ============
    N    = 7
    beta = 0.95

    p = np.array([0.95, 0.90, 0.85, 0.78, 0.70, 0.55, 0.35, 0.0])

    Cm   = 5.0
    Km   = 100.0
    c_sw = 3.0

    # ============ PRINT STATE SPACE ============
    print_state_space(N)

    # ============ BUILD ARM ============
    print("Building single-machine restless bandit arm...")
    t0 = time.time()
    arm = SingleMachineRB(N=N, Cm=Cm, Km=Km, c_sw=c_sw, p=p, beta=beta)
    print(f"  Built in {time.time()-t0:.3f}s\n")

    # ============ QUICK TEST: single VI call ============
    print("Quick test: VI at w=0 ...")
    t0 = time.time()
    V_test, pol_test, iters = arm.vi_discounted(w=0.0)
    print(f"  Converged in {iters} iterations, {time.time()-t0:.4f}s")
    print(f"  V = {V_test}")
    print(f"  Policy (1=active): {pol_test}")
    print()

    # ============ INDEXABILITY CHECK ============
    print("=" * 65)
    print("  INDEXABILITY CHECK")
    print("=" * 65)
    check_indexability(arm)
    print()

    # ============ DISCOUNTED WHITTLE INDICES ============
    print("=" * 65)
    print(f"  DISCOUNTED-COST WHITTLE INDICES  (β = {beta})")
    print("=" * 65)
    t0 = time.time()
    W_disc = whittle_indices_discounted(arm)
    print(f"  Computed in {time.time()-t0:.2f}s")
    print_indices("Discounted Whittle Indices", W_disc, N)

    # ============ AVERAGE-COST WHITTLE INDICES ============
    print("=" * 65)
    print("  AVERAGE-COST WHITTLE INDICES (RVI)")
    print("=" * 65)
    t0 = time.time()

    # Test single RVI call first
    print("  Testing RVI at w=0 ...")
    h_test, pol_test_rvi, g_test, iters_rvi = arm.rvi_average(w=0.0)
    print(f"    Converged in {iters_rvi} iters, g={g_test:.4f}")
    print(f"    Policy: {pol_test_rvi}")

    W_avg = whittle_indices_average(arm)
    print(f"  Computed in {time.time()-t0:.2f}s")
    print_indices("Average-Cost Whittle Indices", W_avg, N)

    # ============ INSTANCE A: M=3, R=1 ============
    M_A, R_A = 3, 1
    arms_A = [arm] * M_A

    print("=" * 65)
    print(f"  INSTANCE A:  M={M_A} machines, R={R_A} repairman")
    print("=" * 65)

    disc_cost_A, avg_cost_A = simulate_whittle_policy(
        arms_A, R_A, [W_disc]*M_A, T=100000, beta=beta)
    print(f"  Discounted-index policy:")
    print(f"    Total discounted cost:   {disc_cost_A:>12.2f}")
    print(f"    Average cost/period:     {avg_cost_A:>12.4f}")

    disc_cost_A2, avg_cost_A2 = simulate_whittle_policy(
        arms_A, R_A, [W_avg]*M_A, T=100000, beta=beta)
    print(f"  Average-cost-index policy:")
    print(f"    Total discounted cost:   {disc_cost_A2:>12.2f}")
    print(f"    Average cost/period:     {avg_cost_A2:>12.4f}")
    print()

    # ============ INSTANCE B: M=4, R=2 ============
    M_B, R_B = 4, 2
    arms_B = [arm] * M_B

    print("=" * 65)
    print(f"  INSTANCE B:  M={M_B} machines, R={R_B} repairmen")
    print("=" * 65)

    disc_cost_B, avg_cost_B = simulate_whittle_policy(
        arms_B, R_B, [W_disc]*M_B, T=100000, beta=beta)
    print(f"  Discounted-index policy:")
    print(f"    Total discounted cost:   {disc_cost_B:>12.2f}")
    print(f"    Average cost/period:     {avg_cost_B:>12.4f}")

    disc_cost_B2, avg_cost_B2 = simulate_whittle_policy(
        arms_B, R_B, [W_avg]*M_B, T=100000, beta=beta)
    print(f"  Average-cost-index policy:")
    print(f"    Total discounted cost:   {disc_cost_B2:>12.2f}")
    print(f"    Average cost/period:     {avg_cost_B2:>12.4f}")
    print()

    # ============ SUMMARY ============
    print("=" * 65)
    print("  SUMMARY TABLE")
    print("=" * 65)
    print(f"  {'Metric':<35s} {'A (3M,1R)':>12s} {'B (4M,2R)':>12s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    print(f"  {'Disc idx → disc cost':<35s} {disc_cost_A:>12.2f} {disc_cost_B:>12.2f}")
    print(f"  {'Disc idx → avg cost/period':<35s} {avg_cost_A:>12.4f} {avg_cost_B:>12.4f}")
    print(f"  {'Avg idx  → avg cost/period':<35s} {avg_cost_A2:>12.4f} {avg_cost_B2:>12.4f}")
    print()


if __name__ == "__main__":
    main()