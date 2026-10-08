"""
Machine Repairman with Switching Cost — Whittle Index Computation
================================================================
M = 3 machines, R = 1 repairman, 8 states (0..7), beta = 0.95
Extended state: (x, a_prev) ∈ {0..7} × {0,1} → 16 states per machine
"""

import numpy as np
import time

# ─── Safe import of markovianbandit ───
BANDIT_AVAILABLE = False
BANDIT_API = None  # will be 'arm_class', 'from_P0P1', 'restless_class', 'whittle_func', or None

try:
    import markovianbandit as bandit

    # Probe the API to figure out what's available
    if hasattr(bandit, 'MarkovianBanditArm'):
        BANDIT_API = 'arm_class'
        BANDIT_AVAILABLE = True
    elif hasattr(bandit, 'restless_bandit_from_P0P1_R0R1'):
        BANDIT_API = 'from_P0P1'
        BANDIT_AVAILABLE = True
    elif hasattr(bandit, 'RestlessBandit'):
        BANDIT_API = 'restless_class'
        BANDIT_AVAILABLE = True
    elif hasattr(bandit, 'whittle_index'):
        BANDIT_API = 'whittle_func'
        BANDIT_AVAILABLE = True
    else:
        # Try to list what IS available
        public = [x for x in dir(bandit) if not x.startswith('_')]
        print(f"  [WARNING] markovianbandit imported but no known API found.")
        print(f"  Available attributes: {public}")
        BANDIT_AVAILABLE = False

except ImportError:
    print("  [WARNING] markovianbandit not installed. "
          "Package-based indices will be skipped.")
    print("  Install with: pip install markovianbandit")


def compute_whittle_via_package(P0, P1, R0, R1, beta=None):
    """
    Attempt to compute Whittle indices using markovianbandit package.
    Returns a numpy array of indices, or None if unavailable.
    """
    if not BANDIT_AVAILABLE:
        return None

    try:
        if BANDIT_API == 'arm_class':
            # markovianbandit >= 0.3 style:
            # MarkovianBanditArm(P_passive, P_active, R_passive, R_active)
            arm_obj = bandit.MarkovianBanditArm(P0, P1, R0, R1)
            if beta is not None and hasattr(arm_obj, 'whittle_indices'):
                indices = arm_obj.whittle_indices(discount=beta)
            elif hasattr(arm_obj, 'whittle_indices'):
                indices = arm_obj.whittle_indices()
            else:
                indices = arm_obj.compute_whittle_indices()
            return np.array(indices)

        elif BANDIT_API == 'from_P0P1':
            model_pkg = bandit.restless_bandit_from_P0P1_R0R1(
                P0=P0, P1=P1, R0=R0, R1=R1
            )
            indices = model_pkg.whittle_indices()
            return np.array(indices)

        elif BANDIT_API == 'restless_class':
            model_pkg = bandit.RestlessBandit(P0=P0, P1=P1, R0=R0, R1=R1)
            indices = model_pkg.whittle_indices()
            return np.array(indices)

        elif BANDIT_API == 'whittle_func':
            indices = bandit.whittle_index(P0, P1, R0, R1)
            return np.array(indices)

    except Exception as e:
        print(f"  [WARNING] markovianbandit call failed: {e}")
        # Try brute-force: iterate over all callables
        try:
            public = [x for x in dir(bandit) if not x.startswith('_')]
            print(f"  Available in markovianbandit: {public}")
        except:
            pass
        return None

    return None


# ─────────────────────────────────────────────────────
#  MachineArm class
# ─────────────────────────────────────────────────────
class MachineArm:
    def __init__(self, N, Cm, Km, c_sw, p, beta):
        self.N     = N
        self.Cm    = Cm
        self.Km    = Km
        self.c_sw  = c_sw
        self.p     = np.array(p, dtype=float)
        self.beta  = beta

        self.states = []
        for x in range(N + 1):
            for ap in (0, 1):
                self.states.append((x, ap))
        self.n   = len(self.states)
        self._idx = {s: i for i, s in enumerate(self.states)}

        self._build_matrices()

    def si(self, x, ap):
        return self._idx[(x, ap)]

    def _build_matrices(self):
        n = self.n
        self.P_a      = np.zeros((n, n))
        self.P_b      = np.zeros((n, n))
        self.c_b      = np.zeros(n)
        self.c_a_base = np.zeros(n)

        for x in range(self.N + 1):
            for ap in (0, 1):
                s = self.si(x, ap)

                # Active → (0,1)
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

    def get_P0_P1_R0_R1(self):
        """
        P0 = passive transition,  P1 = active transition
        R0 = passive reward = -cost_passive
        R1 = active  reward = -cost_active
        """
        P0 = self.P_b.copy()
        P1 = self.P_a.copy()
        R0 = -self.c_b.copy()
        R1 = -self.c_a_base.copy()

        # normalise rows (safety)
        for i in range(self.n):
            s0 = P0[i].sum()
            s1 = P1[i].sum()
            if s0 > 0:
                P0[i] /= s0
            if s1 > 0:
                P1[i] /= s1

        return P0, P1, R0, R1

    def vi_discounted(self, w, tol=1e-6, maxiter=50000):
        """
        Value iteration with SPAN SEMINORM stopping.
        """
        c_a  = self.c_a_base + w
        c_b  = self.c_b
        beta = self.beta
        V    = np.zeros(self.n)

        for it in range(1, maxiter + 1):
            Qa   = c_a + beta * (self.P_a @ V)
            Qb   = c_b + beta * (self.P_b @ V)
            Vnew = np.minimum(Qa, Qb)

            diff = Vnew - V
            span = np.max(diff) - np.min(diff)
            V    = Vnew

            if span < tol:
                break

        Qa     = c_a + beta * (self.P_a @ V)
        Qb     = c_b + beta * (self.P_b @ V)
        policy = (Qa <= Qb).astype(int)
        return V, policy, it

    def rvi_average(self, w, tol=1e-6, maxiter=50000):
        """Relative value iteration for average cost."""
        c_a = self.c_a_base + w
        c_b = self.c_b
        h   = np.zeros(self.n)
        g   = 0.0

        for it in range(1, maxiter + 1):
            Qa    = c_a + self.P_a @ h
            Qb    = c_b + self.P_b @ h
            h_new = np.minimum(Qa, Qb)

            g     = h_new[0]
            h_new = h_new - g

            diff = h_new - h
            span = np.max(diff) - np.min(diff)
            h    = h_new

            if span < tol:
                break

        Qa     = c_a + self.P_a @ h
        Qb     = c_b + self.P_b @ h
        policy = (Qa <= Qb).astype(int)
        return h, policy, g, it


# ─────────────────────────────────────────────────────
#  Binary-search Whittle index computation
# ─────────────────────────────────────────────────────
def compute_whittle_disc(arm, wlo=None, whi=None, tol=1e-4):
    """Discounted Whittle index via binary search."""
    if wlo is None:
        wlo = -2.0 * arm.Km
    if whi is None:
        whi = 2.0 * arm.Km

    idx = {}

    _, pol_lo, it_lo = arm.vi_discounted(wlo)
    _, pol_hi, it_hi = arm.vi_discounted(whi)

    print(f"      Boundary check: w_lo={wlo:.1f} (iters={it_lo}), "
          f"w_hi={whi:.1f} (iters={it_hi})")
    print(f"      Active states at w_lo: {np.sum(pol_lo)}, "
          f"at w_hi: {np.sum(pol_hi)}")

    for si_idx in range(arm.n):
        st = arm.states[si_idx]

        if pol_lo[si_idx] == 0:
            idx[st] = wlo
            continue
        if pol_hi[si_idx] == 1:
            idx[st] = whi
            continue

        lo, hi = wlo, whi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            _, pol, _ = arm.vi_discounted(mid)
            if pol[si_idx] == 1:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol:
                break

        idx[st] = 0.5 * (lo + hi)

    return idx


def compute_whittle_avg(arm, wlo=None, whi=None, tol=1e-4):
    """Average-cost Whittle index via binary search."""
    if wlo is None:
        wlo = -2.0 * arm.Km
    if whi is None:
        whi = 2.0 * arm.Km

    idx = {}

    _, pol_lo, _, it_lo = arm.rvi_average(wlo)
    _, pol_hi, _, it_hi = arm.rvi_average(whi)

    print(f"      Boundary check: w_lo={wlo:.1f} (iters={it_lo}), "
          f"w_hi={whi:.1f} (iters={it_hi})")
    print(f"      Active states at w_lo: {np.sum(pol_lo)}, "
          f"at w_hi: {np.sum(pol_hi)}")

    for si_idx in range(arm.n):
        st = arm.states[si_idx]

        if pol_lo[si_idx] == 0:
            idx[st] = wlo
            continue
        if pol_hi[si_idx] == 1:
            idx[st] = whi
            continue

        lo, hi = wlo, whi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            _, pol, _, _ = arm.rvi_average(mid)
            if pol[si_idx] == 1:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol:
                break

        idx[st] = 0.5 * (lo + hi)

    return idx


# ─────────────────────────────────────────────────────
#  Parameter generation & printing
# ─────────────────────────────────────────────────────
def generate_parameters(M, N, D, alpha, rng):
    machines = []
    for m in range(M):
        raw = rng.uniform(0, 1, size=N)
        raw = np.sort(raw)[::-1]
        p   = np.append(raw, 0.0)
        Cm  = rng.uniform(D, D + 25)
        Km  = alpha * Cm
        machines.append({'p': p, 'Cm': Cm, 'Km': Km})
    return machines


def print_machine_params(m_id, params):
    print(f"    Machine {m_id + 1}:")
    print(f"      Cm = {params['Cm']:.4f},  Km = {params['Km']:.4f}")
    p_str = ', '.join([f"{v:.4f}" for v in params['p']])
    print(f"      p  = [{p_str}]")


def print_whittle_table(title, W, N):
    print(f"\n    {title}")
    print(f"    {'x':>3s}  {'W(x,0)[passive]':>16s}  {'W(x,1)[active]':>16s}  {'Diff(1-0)':>12s}")
    print(f"    {'---':>3s}  {'---------------':>16s}  {'--------------':>16s}  {'---------':>12s}")
    for x in range(N + 1):
        w0 = W.get((x, 0), float('nan'))
        w1 = W.get((x, 1), float('nan'))
        d  = w1 - w0
        print(f"    {x:>3d}  {w0:>16.6f}  {w1:>16.6f}  {d:>12.6f}")


def print_package_comparison(arm, m_label=""):
    """
    Compare binary-search Whittle indices with markovianbandit package.
    """
    P0, P1, R0, R1 = arm.get_P0_P1_R0_R1()

    pkg_indices = compute_whittle_via_package(P0, P1, R0, R1, beta=arm.beta)

    if pkg_indices is None:
        print(f"    {m_label} Package comparison: SKIPPED (package unavailable)")
        return None

    print(f"\n    {m_label} Package Whittle Indices:")
    print(f"    {'State':>12s}  {'Binary-Search':>14s}  {'Package':>14s}  {'Diff':>12s}")
    print(f"    {'-'*12}  {'-'*14}  {'-'*14}  {'-'*12}")

    # Also compute binary-search for comparison (discounted)
    bs_indices = compute_whittle_disc(arm)

    for i, (x, ap) in enumerate(arm.states):
        bs_val  = bs_indices.get((x, ap), float('nan'))
        pkg_val = pkg_indices[i] if i < len(pkg_indices) else float('nan')
        diff    = pkg_val - bs_val
        print(f"    {f'({x},{ap})':>12s}  {bs_val:>14.6f}  {pkg_val:>14.6f}  {diff:>12.6f}")

    return pkg_indices


# ─────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────
def main():
    M    = 3
    R    = 1
    N    = 7
    beta = 0.95
    c_sw = 3.0

    D     = 10.0
    alpha = 3.0
    seed  = 42

    NUM_SETS = 5

    print("=" * 70)
    print("  MACHINE REPAIRMAN WITH SWITCHING COST")
    print("=" * 70)
    print(f"  M={M}, R={R}, N={N}, beta={beta}, c_sw={c_sw}")
    print(f"  (D, alpha) = ({D}, {alpha})")
    print(f"  Random parameter sets = {NUM_SETS}")
    print(f"  markovianbandit available: {BANDIT_AVAILABLE} (API: {BANDIT_API})")

    # State space
    print(f"\n  Extended states per machine: {2*(N+1)}")
    print(f"  Passive family (a_prev=0): {[(x,0) for x in range(N+1)]}")
    print(f"  Active  family (a_prev=1): {[(x,1) for x in range(N+1)]}")

    # ══════ CONVERGENCE TEST ══════
    print(f"\n{'='*70}")
    print("  CONVERGENCE TEST (Prob Matrix given manually)")
    print(f"{'='*70}")

    test_p   = np.array([0.95, 0.85, 0.75, 0.65, 0.5, 0.35, 0.15, 0.0])
    test_arm = MachineArm(N=N, Cm=10.0, Km=30.0, c_sw=c_sw,
                          p=test_p, beta=beta)

    print("\n  Discounted VI convergence:")
    for test_w in [-50.0, -10.0, 0.0, 10.0, 50.0]:
        t0 = time.time()
        V, pol, iters = test_arm.vi_discounted(test_w)
        dt = time.time() - t0
        print(f"  w={test_w:>7.1f}: iters={iters:>5d}, "
              f"time={dt:.4f}s, active={np.sum(pol):>2d}, "
              f"V_range=[{V.min():.1f}, {V.max():.1f}]")

    print("\n  Average-cost RVI convergence:")
    for test_w in [-50.0, -10.0, 0.0, 10.0, 50.0]:
        t0 = time.time()
        h, pol, g, iters = test_arm.rvi_average(test_w)
        dt = time.time() - t0
        print(f"  w={test_w:>7.1f}: iters={iters:>5d}, "
              f"time={dt:.4f}s, active={np.sum(pol):>2d}, "
              f"g={g:.4f}")

    # Test Whittle index on test arm
    print("\n  Computing test Whittle indices (discounted)...")
    t0 = time.time()
    W_test = compute_whittle_disc(test_arm)
    print(f"  Done in {time.time()-t0:.2f}s")
    print_whittle_table("Test Arm — Discounted Whittle Indices", W_test, N)

    print("\n  Computing test Whittle indices (average cost)...")
    t0 = time.time()
    W_test_avg = compute_whittle_avg(test_arm)
    print(f"  Done in {time.time()-t0:.2f}s")
    print_whittle_table("Test Arm — Average-Cost Whittle Indices", W_test_avg, N)

    # Package comparison on test arm
    print("\n  Comparing with markovianbandit package (test arm)...")
    print_package_comparison(test_arm, m_label="Test Arm")

    # ══════ PACKAGE COMPARISON ON GENERATED MACHINES ══════
    print(f"\n{'='*70}")
    print("  PACKAGE COMPARISON ON GENERATED MACHINES")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    machine_params = generate_parameters(M, N, D, alpha, rng)

    print(f"\n  Generated parameters for {M} machines (N={N}, D={D}, "
          f"alpha={alpha}, seed={seed}):")
    for m_id, params in enumerate(machine_params):
        print_machine_params(m_id, params)

    for m_id, params in enumerate(machine_params):
        print(f"\n  --- Machine {m_id + 1} ---")

        arm = MachineArm(
            N=N, Cm=params['Cm'], Km=params['Km'],
            c_sw=c_sw, p=params['p'], beta=beta,
        )

        P0, P1, R0, R1 = arm.get_P0_P1_R0_R1()
        print(f"  States: {arm.n},  Shapes: P0{P0.shape}  P1{P1.shape}  "
              f"R0{R0.shape}  R1{R1.shape}")

        # Package indices
        pkg_indices = compute_whittle_via_package(P0, P1, R0, R1, beta=beta)

        if pkg_indices is not None:
            print(f"  Whittle indices (package):")
            for i, (x, ap) in enumerate(arm.states):
                val = pkg_indices[i] if i < len(pkg_indices) else float('nan')
                print(f"    state {i:2d}  (x={x}, ap={ap})  →  WI = {val:.6f}")
        else:
            print(f"  Whittle indices (package): SKIPPED")

        # Binary-search indices for comparison
        W_d = compute_whittle_disc(arm)
        print_whittle_table(f"Machine {m_id+1} — Binary-Search Discounted", W_d, N)

    # ══════ MAIN EXPERIMENT ══════
    all_disc = []
    all_avg  = []

    for s in range(NUM_SETS):
        rng = np.random.default_rng(seed=1000 + s)

        print(f"\n{'='*70}")
        print(f"  PARAMETER SET {s+1}/{NUM_SETS}")
        print(f"{'='*70}")

        params_list = generate_parameters(M, N, D, alpha, rng)
        for m in range(M):
            print_machine_params(m, params_list[m])

        arms = []
        for m in range(M):
            arm = MachineArm(
                N=N, Cm=params_list[m]['Cm'], Km=params_list[m]['Km'],
                c_sw=c_sw, p=params_list[m]['p'], beta=beta
            )
            arms.append(arm)

        disc_indices = []
        avg_indices  = []

        for m in range(M):
            print(f"\n  Machine {m+1}: Computing Whittle indices...")

            t0  = time.time()
            W_d = compute_whittle_disc(arms[m])
            t1  = time.time()
            W_a = compute_whittle_avg(arms[m])
            t2  = time.time()

            print(f"    Times: disc={t1-t0:.2f}s, avg={t2-t1:.2f}s")

            print_whittle_table(
                f"Machine {m+1} DISCOUNTED (beta={beta})", W_d, N)
            print_whittle_table(
                f"Machine {m+1} AVERAGE COST", W_a, N)

            disc_indices.append(W_d)
            avg_indices.append(W_a)

        all_disc.append(disc_indices)
        all_avg.append(avg_indices)

        # ── Policy decisions ──
        print(f"\n  POLICY DECISIONS (discounted indices):")
        print(f"  {'Joint State':>45s}  {'W-values':>30s}  {'Decision':>10s}")
        print(f"  {'-'*45}  {'-'*30}  {'-'*10}")

        ex_rng = np.random.default_rng(seed=42 + s)
        for _ in range(16):
            jx  = ex_rng.integers(0, N + 1, size=M)
            jap = ex_rng.integers(0, 2, size=M)
            wv  = [disc_indices[m][(jx[m], jap[m])] for m in range(M)]
            best = int(np.argmax(wv))
            idle = all(w < 0 for w in wv)

            st_str = "  ".join(
                [f"M{m+1}=({jx[m]},{jap[m]})" for m in range(M)])
            wv_str = "  ".join([f"{w:.2f}" for w in wv])
            dec    = "IDLE" if idle else f"M{best+1}"
            print(f"  {st_str:>45s}  {wv_str:>30s}  {dec:>10s}")

    # ══════ SUMMARY ══════
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY: DISCOUNTED WHITTLE INDICES")
    print(f"  ({NUM_SETS} sets × {M} machines = {NUM_SETS*M} instances)")
    print(f"{'='*70}")
    print(f"  {'State':>10s}  {'Min':>10s}  {'Q1':>10s}  "
          f"{'Median':>10s}  {'Q3':>10s}  {'Max':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}")

    for x in range(N + 1):
        for ap in (0, 1):
            vals = []
            for s_idx in range(NUM_SETS):
                for m in range(M):
                    vals.append(all_disc[s_idx][m][(x, ap)])
            vals = np.array(vals)
            fam = 'P' if ap == 0 else 'A'
            q1  = np.percentile(vals, 25)
            med = np.percentile(vals, 50)
            q3  = np.percentile(vals, 75)
            print(f"  {f'({x},{ap})[{fam}]':>10s}  {vals.min():>10.4f}  "
                  f"{q1:>10.4f}  {med:>10.4f}  {q3:>10.4f}  {vals.max():>10.4f}")

    print(f"\n{'='*70}")
    print(f"  SUMMARY: AVERAGE-COST WHITTLE INDICES")
    print(f"{'='*70}")
    print(f"  {'State':>10s}  {'Min':>10s}  {'Q1':>10s}  "
          f"{'Median':>10s}  {'Q3':>10s}  {'Max':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}")

    for x in range(N + 1):
        for ap in (0, 1):
            vals = []
            for s_idx in range(NUM_SETS):
                for m in range(M):
                    vals.append(all_avg[s_idx][m][(x, ap)])
            vals = np.array(vals)
            fam = 'P' if ap == 0 else 'A'
            q1  = np.percentile(vals, 25)
            med = np.percentile(vals, 50)
            q3  = np.percentile(vals, 75)
            print(f"  {f'({x},{ap})[{fam}]':>10s}  {vals.min():>10.4f}  "
                  f"{q1:>10.4f}  {med:>10.4f}  {q3:>10.4f}  {vals.max():>10.4f}")

    # ── Switching cost effect ──
    print(f"\n{'='*70}")
    print(f"  SWITCHING COST EFFECT: W(x,1) - W(x,0)")
    print(f"  Expected: ≈ c_sw = {c_sw} (discounted may differ)")
    print(f"{'='*70}")

    print(f"\n  {'x':>3s}  {'Disc mean diff':>15s}  {'Avg mean diff':>15s}")
    print(f"  {'---':>3s}  {'---------------':>15s}  {'---------------':>15s}")
    for x in range(N + 1):
        d_disc = []
        d_avg  = []
        for s_idx in range(NUM_SETS):
            for m in range(M):
                d_disc.append(
                    all_disc[s_idx][m][(x,1)] - all_disc[s_idx][m][(x,0)])
                d_avg.append(
                    all_avg[s_idx][m][(x,1)] - all_avg[s_idx][m][(x,0)])
        print(f"  {x:>3d}  {np.mean(d_disc):>15.6f}  {np.mean(d_avg):>15.6f}")

    print(f"\n{'='*70}")
    print("  DONE.")
    print(f"{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total runtime: {time.time()-t0:.2f}s")