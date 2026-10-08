"""
Machine Repairman with Switching Cost — Whittle Index Computation
================================================================
Setup:
  - M = 3 machines, R = 1 repairman
  - 8 damage states per machine (0..7), N = 7
  - Extended state: (x, a_prev) ∈ {0..7} × {0,1} → 16 states
  - 1 (D, alpha) pair, 5 random parameter sets
  - Switching cost c_sw incurred when repairman moves to a new machine

Parameter generation (following the paper):
  - Survival probs p[0] >= p[1] >= ... >= p[6] >= p[7]=0
    drawn from U[0,1], sorted descending
  - Cm ~ U(D, D+25),  Km = alpha * Cm
  - beta = 0.95
"""

import numpy as np
import time

# ──────────────────────────────────────────────
#  Single Machine Restless Bandit Arm
# ──────────────────────────────────────────────

class MachineArm:
    """
    One machine with extended state (x, a_prev).
    Precomputes transition matrices once.
    """

    def __init__(self, N, Cm, Km, c_sw, p, beta):
        self.N     = N
        self.Cm    = Cm
        self.Km    = Km
        self.c_sw  = c_sw
        self.p     = np.array(p, dtype=float)
        self.beta  = beta

        # State list: (x, a_prev)
        self.states = []
        for x in range(N + 1):
            for ap in (0, 1):
                self.states.append((x, ap))
        self.n   = len(self.states)   # 2*(N+1) = 16
        self._idx = {s: i for i, s in enumerate(self.states)}

        # Precompute
        self._build_matrices()

    def si(self, x, ap):
        return self._idx[(x, ap)]

    def _build_matrices(self):
        n = self.n
        self.P_a = np.zeros((n, n))
        self.P_b = np.zeros((n, n))
        self.c_b = np.zeros(n)
        self.c_a_base = np.zeros(n)

        for x in range(self.N + 1):
            for ap in (0, 1):
                s = self.si(x, ap)

                # Active: repair → (0,1)
                self.P_a[s, self.si(0, 1)] = 1.0
                self.c_a_base[s] = self.Cm + (self.c_sw if ap == 0 else 0.0)

                # Passive
                px = self.p[x] if x < self.N else 0.0
                self.c_b[s] = self.Km * (1.0 - px)

                if x < self.N:
                    self.P_b[s, self.si(x + 1, 0)] = px
                    self.P_b[s, self.si(0, 0)]     = 1.0 - px
                else:
                    self.P_b[s, self.si(0, 0)] = 1.0

    def vi_discounted(self, w, tol=1e-8, maxiter=10000):
        """Value iteration for discounted cost with subsidy w."""
        c_a = self.c_a_base + w
        V = np.zeros(self.n)
        beta = self.beta

        for it in range(1, maxiter + 1):
            Qa = c_a + beta * (self.P_a @ V)
            Qb = self.c_b + beta * (self.P_b @ V)
            Vnew = np.minimum(Qa, Qb)
            if np.max(np.abs(Vnew - V)) < tol:
                V = Vnew
                break
            V = Vnew

        Qa = c_a + beta * (self.P_a @ V)
        Qb = self.c_b + beta * (self.P_b @ V)
        policy = (Qa <= Qb).astype(int)
        return V, policy

    def rvi_average(self, w, tol=1e-8, maxiter=100000):
        """Relative value iteration for average cost with subsidy w."""
        c_a = self.c_a_base + w
        h = np.zeros(self.n)
        g = 0.0

        for it in range(1, maxiter + 1):
            Qa = c_a + self.P_a @ h
            Qb = self.c_b + self.P_b @ h
            h_new = np.minimum(Qa, Qb)
            g = h_new[0]
            h_new = h_new - g

            diff = h_new - h
            span = np.max(diff) - np.min(diff)
            if span < tol:
                h = h_new
                break
            h = h_new

        Qa = c_a + self.P_a @ h
        Qb = self.c_b + self.P_b @ h
        policy = (Qa <= Qb).astype(int)
        return h, policy, g


# ──────────────────────────────────────────────
#  Whittle Index via Binary Search
# ──────────────────────────────────────────────

def compute_whittle_disc(arm, wlo=-300, whi=300, tol=1e-5):
    """Compute discounted Whittle index for every extended state."""
    idx = {}
    _, pol_lo = arm.vi_discounted(wlo)
    _, pol_hi = arm.vi_discounted(whi)

    for si in range(arm.n):
        st = arm.states[si]
        if pol_lo[si] == 0:
            idx[st] = wlo
            continue
        if pol_hi[si] == 1:
            idx[st] = whi
            continue
        lo, hi = wlo, whi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            _, pol = arm.vi_discounted(mid)
            if pol[si] == 1:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        idx[st] = 0.5 * (lo + hi)
    return idx


def compute_whittle_avg(arm, wlo=-300, whi=300, tol=1e-5):
    """Compute average-cost Whittle index for every extended state."""
    idx = {}
    _, pol_lo, _ = arm.rvi_average(wlo)
    _, pol_hi, _ = arm.rvi_average(whi)

    for si in range(arm.n):
        st = arm.states[si]
        if pol_lo[si] == 0:
            idx[st] = wlo
            continue
        if pol_hi[si] == 1:
            idx[st] = whi
            continue
        lo, hi = wlo, whi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            _, pol, _ = arm.rvi_average(mid)
            if pol[si] == 1:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        idx[st] = 0.5 * (lo + hi)
    return idx


# ──────────────────────────────────────────────
#  Parameter Generation (Following the Paper)
# ──────────────────────────────────────────────

def generate_parameters(M, N, D, alpha, rng):
    """
    Generate random parameters for M machines.

    For each machine m:
      - p_m = 7 samples from U[0,1], sorted descending, p[N]=0
      - Cm ~ U(D, D+25)
      - Km = alpha * Cm

    Returns list of dicts, one per machine.
    """
    machines = []
    for m in range(M):
        # Survival probabilities
        raw = rng.uniform(0, 1, size=N)        # 7 values
        raw = np.sort(raw)[::-1]               # sort descending
        p = np.append(raw, 0.0)                # p[7] = 0

        # Costs
        Cm = rng.uniform(D, D + 25)
        Km = alpha * Cm

        machines.append({
            'p': p,
            'Cm': Cm,
            'Km': Km,
        })
    return machines


# ──────────────────────────────────────────────
#  Display
# ──────────────────────────────────────────────

def print_machine_params(m_id, params):
    """Print one machine's parameters."""
    print(f"    Machine {m_id + 1}:")
    print(f"      Cm = {params['Cm']:.4f}")
    print(f"      Km = {params['Km']:.4f}")
    p_str = ', '.join([f"{v:.4f}" for v in params['p']])
    print(f"      p  = [{p_str}]")


def print_whittle_table(title, W, N):
    """Print Whittle indices in a neat table."""
    print(f"\n    {title}")
    print(f"    {'x':>3s}  {'W(x,0) passive':>16s}  {'W(x,1) active':>16s}")
    print(f"    {'---':>3s}  {'----------------':>16s}  {'----------------':>16s}")
    for x in range(N + 1):
        w0 = W.get((x, 0), float('nan'))
        w1 = W.get((x, 1), float('nan'))
        print(f"    {x:>3d}  {w0:>16.6f}  {w1:>16.6f}")


# ──────────────────────────────────────────────
#  MAIN EXPERIMENT
# ──────────────────────────────────────────────

def main():
    # ═══════ FIXED SETTINGS ═══════
    M    = 3       # machines
    R    = 1       # repairman
    N    = 7       # states 0..7
    beta = 0.95
    c_sw = 3.0     # switching cost

    # One (D, alpha) pair
    D     = 10.0
    alpha = 3.0

    # Number of random parameter sets
    NUM_RANDOM_SETS = 5

    print("=" * 70)
    print("  MACHINE REPAIRMAN WITH SWITCHING COST — WHITTLE INDEX STUDY")
    print("=" * 70)
    print(f"  M = {M} machines,  R = {R} repairman")
    print(f"  N = {N} (states 0..{N}),  beta = {beta}")
    print(f"  Switching cost c_sw = {c_sw}")
    print(f"  (D, alpha) = ({D}, {alpha})")
    print(f"  Number of random parameter sets = {NUM_RANDOM_SETS}")

    # State space info
    print(f"\n  Extended state space per machine:")
    print(f"    (x, a_prev) ∈ {{0,...,{N}}} × {{0,1}}")
    print(f"    Total states per machine: {2 * (N + 1)}")
    print(f"    Passive family: ", [(x, 0) for x in range(N + 1)])
    print(f"    Active family:  ", [(x, 1) for x in range(N + 1)])

    # ═══════ MAIN LOOP OVER RANDOM SETS ═══════
    all_disc_indices = []   # store for summary
    all_avg_indices  = []

    for s in range(NUM_RANDOM_SETS):
        rng = np.random.default_rng(seed=1000 + s)

        print(f"\n{'='*70}")
        print(f"  RANDOM PARAMETER SET {s + 1} / {NUM_RANDOM_SETS}")
        print(f"{'='*70}")

        # Generate parameters
        params_list = generate_parameters(M, N, D, alpha, rng)

        # Print parameters
        print(f"\n  Parameters (D={D}, alpha={alpha}):")
        for m in range(M):
            print_machine_params(m, params_list[m])

        # Build arms
        arms = []
        for m in range(M):
            arm = MachineArm(
                N    = N,
                Cm   = params_list[m]['Cm'],
                Km   = params_list[m]['Km'],
                c_sw = c_sw,
                p    = params_list[m]['p'],
                beta = beta
            )
            arms.append(arm)

        # ──── Compute Whittle Indices ────
        disc_indices = []
        avg_indices  = []

        for m in range(M):
            print(f"\n  Computing Whittle indices for Machine {m + 1}...")
            t0 = time.time()

            # Discounted
            W_disc = compute_whittle_disc(arms[m])
            t1 = time.time()

            # Average cost
            W_avg = compute_whittle_avg(arms[m])
            t2 = time.time()

            print(f"    Discounted: {t1 - t0:.3f}s,  Average: {t2 - t1:.3f}s")

            # Print tables
            print_whittle_table(
                f"Machine {m+1} — DISCOUNTED Whittle Indices (β={beta})",
                W_disc, N)
            print_whittle_table(
                f"Machine {m+1} — AVERAGE-COST Whittle Indices",
                W_avg, N)

            disc_indices.append(W_disc)
            avg_indices.append(W_avg)

        all_disc_indices.append(disc_indices)
        all_avg_indices.append(avg_indices)

        # ──── Show Whittle Index Policy for All States ────
        print(f"\n  {'─'*60}")
        print(f"  WHITTLE INDEX POLICY ILLUSTRATION (Discounted)")
        print(f"  R={R} repairman: activate the machine with HIGHEST index")
        print(f"  {'─'*60}")

        # Show a few example joint states
        print(f"\n  Example joint states and decisions:")
        print(f"  {'Machine States':>40s}  {'Indices':>35s}  {'Activate':>10s}")
        print(f"  {'-'*40}  {'-'*35}  {'-'*10}")

        example_rng = np.random.default_rng(seed=42 + s)
        for _ in range(8):
            # Random joint state
            joint_x  = example_rng.integers(0, N + 1, size=M)
            joint_ap = example_rng.integers(0, 2, size=M)

            # Get indices
            w_vals = []
            for m in range(M):
                w_vals.append(disc_indices[m][(joint_x[m], joint_ap[m])])

            # Decision: activate machine with highest index
            best_m = np.argmax(w_vals)

            # Check if all indices negative → idle repairman
            idle = all(w < 0 for w in w_vals)

            states_str = "  ".join(
                [f"M{m+1}=({joint_x[m]},{joint_ap[m]})" for m in range(M)])
            idx_str = "  ".join([f"{w:.2f}" for w in w_vals])

            if idle:
                dec = "IDLE"
            else:
                dec = f"M{best_m + 1}"

            print(f"  {states_str:>40s}  {idx_str:>35s}  {dec:>10s}")

    # ═══════ SUMMARY ACROSS ALL PARAMETER SETS ═══════
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: WHITTLE INDICES ACROSS {NUM_RANDOM_SETS} PARAMETER SETS")
    print(f"  (D={D}, alpha={alpha}, c_sw={c_sw}, beta={beta})")
    print(f"{'='*70}")

    # For each state, collect indices across all sets and machines
    print(f"\n  DISCOUNTED COST — Statistics over {NUM_RANDOM_SETS * M} machine instances")
    print(f"  {'State':>10s}  {'Min':>10s}  {'Mean':>10s}  {'Max':>10s}  {'Std':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for x in range(N + 1):
        for ap in (0, 1):
            vals = []
            for s in range(NUM_RANDOM_SETS):
                for m in range(M):
                    vals.append(all_disc_indices[s][m][(x, ap)])
            vals = np.array(vals)
            fam = 'P' if ap == 0 else 'A'
            print(f"  {f'({x},{ap})[{fam}]':>10s}  "
                  f"{vals.min():>10.4f}  {vals.mean():>10.4f}  "
                  f"{vals.max():>10.4f}  {vals.std():>10.4f}")

    print(f"\n  AVERAGE COST — Statistics over {NUM_RANDOM_SETS * M} machine instances")
    print(f"  {'State':>10s}  {'Min':>10s}  {'Mean':>10s}  {'Max':>10s}  {'Std':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for x in range(N + 1):
        for ap in (0, 1):
            vals = []
            for s in range(NUM_RANDOM_SETS):
                for m in range(M):
                    vals.append(all_avg_indices[s][m][(x, ap)])
            vals = np.array(vals)
            fam = 'P' if ap == 0 else 'A'
            print(f"  {f'({x},{ap})[{fam}]':>10s}  "
                  f"{vals.min():>10.4f}  {vals.mean():>10.4f}  "
                  f"{vals.max():>10.4f}  {vals.std():>10.4f}")

    # ──── Effect of Switching Cost ────
    print(f"\n{'='*70}")
    print(f"  SWITCHING COST EFFECT: Comparing W(x,0) vs W(x,1)")
    print(f"  Difference = W(x,1) - W(x,0) should be ≈ c_sw = {c_sw}")
    print(f"{'='*70}")

    print(f"\n  DISCOUNTED:")
    print(f"  {'x':>3s}  {'W(x,1)-W(x,0) values across all instances...':>50s}")
    for x in range(N + 1):
        diffs = []
        for s in range(NUM_RANDOM_SETS):
            for m in range(M):
                d = all_disc_indices[s][m][(x, 1)] - all_disc_indices[s][m][(x, 0)]
                diffs.append(d)
        diffs = np.array(diffs)
        diff_str = ', '.join([f"{d:.4f}" for d in diffs[:6]])
        print(f"  {x:>3d}  mean={diffs.mean():.4f}  "
              f"std={diffs.std():.4f}  [{diff_str}, ...]")

    print(f"\n  AVERAGE COST:")
    for x in range(N + 1):
        diffs = []
        for s in range(NUM_RANDOM_SETS):
            for m in range(M):
                d = all_avg_indices[s][m][(x, 1)] - all_avg_indices[s][m][(x, 0)]
                diffs.append(d)
        diffs = np.array(diffs)
        diff_str = ', '.join([f"{d:.4f}" for d in diffs[:6]])
        print(f"  {x:>3d}  mean={diffs.mean():.4f}  "
              f"std={diffs.std():.4f}  [{diff_str}, ...]")

    print(f"\n{'='*70}")
    print(f"  DONE.")
    print(f"{'='*70}")


if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f"\n  Total runtime: {time.time() - t_start:.2f}s")