"""
Machine Repairman with Switching Cost — Whittle Index Computation
================================================================
Uses POLICY ITERATION (guaranteed finite convergence) instead of
Value Iteration (which can stall near indifference points).
"""

import numpy as np
import time


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
        self.P_a = np.zeros((n, n))
        self.P_b = np.zeros((n, n))
        self.c_b = np.zeros(n)
        self.c_a_base = np.zeros(n)

        for x in range(self.N + 1):
            for ap in (0, 1):
                s = self.si(x, ap)

                # Active → (0,1) deterministically
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

    # ═══════════════════════════════════════════════════
    #  DISCOUNTED POLICY ITERATION  
    # ═══════════════════════════════════════════════════
    def pi_discounted(self, w, maxiter=200):
        """
        Policy Iteration for discounted cost.
        Each iteration:
          1) Policy Evaluation: solve (I - beta*P_pi) V = c_pi  (linear system)
          2) Policy Improvement: pick best action per state
        Guaranteed to converge in at most 2^n iterations (usually <10).
        """
        n    = self.n
        beta = self.beta
        c_a  = self.c_a_base + w
        c_b  = self.c_b
        P_a  = self.P_a
        P_b  = self.P_b
        I    = np.eye(n)

        # Initial policy: passive everywhere
        policy = np.zeros(n, dtype=int)

        for it in range(1, maxiter + 1):
            # ── Policy Evaluation: solve linear system ──
            # Build P_pi and c_pi from current policy
            P_pi = np.zeros((n, n))
            c_pi = np.zeros(n)
            for s in range(n):
                if policy[s] == 1:   # active
                    P_pi[s, :] = P_a[s, :]
                    c_pi[s]    = c_a[s]
                else:                # passive
                    P_pi[s, :] = P_b[s, :]
                    c_pi[s]    = c_b[s]

            # Solve: (I - beta * P_pi) V = c_pi
            A = I - beta * P_pi
            V = np.linalg.solve(A, c_pi)

            # ── Policy Improvement ──
            Qa = c_a + beta * (P_a @ V)
            Qb = c_b + beta * (P_b @ V)
            new_policy = (Qa <= Qb).astype(int)

            if np.array_equal(new_policy, policy):
                # Policy stable → optimal
                return V, policy, it

            policy = new_policy

        return V, policy, maxiter

    # ═══════════════════════════════════════════════════
    #  AVERAGE-COST POLICY ITERATION
    # ═══════════════════════════════════════════════════
    def pi_average(self, w, maxiter=200):
        """
        Policy Iteration for average cost.
        Each iteration:
          1) Solve: g*e + h = c_pi + P_pi * h  (with h[0]=0 as reference)
             Rewrite: (I - P_pi) h + g*e = c_pi
             Fix h[0] = 0, solve the system.
          2) Policy Improvement.
        """
        n   = self.n
        c_a = self.c_a_base + w
        c_b = self.c_b
        P_a = self.P_a
        P_b = self.P_b

        # Initial policy: passive everywhere
        policy = np.zeros(n, dtype=int)

        for it in range(1, maxiter + 1):
            # ── Policy Evaluation ──
            P_pi = np.zeros((n, n))
            c_pi = np.zeros(n)
            for s in range(n):
                if policy[s] == 1:
                    P_pi[s, :] = P_a[s, :]
                    c_pi[s]    = c_a[s]
                else:
                    P_pi[s, :] = P_b[s, :]
                    c_pi[s]    = c_b[s]

            # Solve: (I - P_pi) h + g * e = c_pi, with h[0] = 0
            # System of n equations, n unknowns: h[1],...,h[n-1], g
            #
            # For row s:  h[s] - sum_j P_pi[s,j]*h[j] + g = c_pi[s]
            #
            # Build (n x n) system: columns = [h[1], h[2], ..., h[n-1], g]
            # Row 0 uses h[0]=0:  -sum_{j>=1} P[0,j]*h[j] + g = c[0] - P[0,0]*0
            # Row s (s>=1): h[s] - sum_j P[s,j]*h[j] + g = c[s]

            B = np.zeros((n, n))
            rhs = np.zeros(n)

            IminusP = np.eye(n) - P_pi

            for s in range(n):
                # h[0] = 0 → move column 0 of (I-P) to rhs
                for j in range(1, n):
                    B[s, j - 1] = IminusP[s, j]
                B[s, n - 1] = 1.0      # coefficient of g
                rhs[s] = c_pi[s] - IminusP[s, 0] * 0.0  # h[0]=0

            sol = np.linalg.solve(B, rhs)
            h = np.zeros(n)
            h[1:] = sol[:n - 1]
            h[0]  = 0.0
            g     = sol[n - 1]

            # ── Policy Improvement ──
            Qa = c_a + P_a @ h
            Qb = c_b + P_b @ h
            new_policy = (Qa <= Qb).astype(int)

            if np.array_equal(new_policy, policy):
                return h, policy, g, it

            policy = new_policy

        return h, policy, g, maxiter


# ──────────────────────────────────────────────
#  Whittle Index Computation
# ──────────────────────────────────────────────

def compute_whittle_disc(arm, wlo=None, whi=None, tol=1e-4):
    if wlo is None:
        wlo = -(arm.Km + arm.Cm + arm.c_sw + 10.0)
    if whi is None:
        whi = (arm.Km + arm.Cm + arm.c_sw + 10.0)

    idx = {}
    _, pol_lo, it_lo = arm.pi_discounted(wlo)
    _, pol_hi, it_hi = arm.pi_discounted(whi)

    print(f"      Bounds: [{wlo:.1f}, {whi:.1f}]  "
          f"iters: lo={it_lo}, hi={it_hi}  "
          f"active: lo={np.sum(pol_lo)}, hi={np.sum(pol_hi)}")

    for si in range(arm.n):
        st = arm.states[si]
        if pol_lo[si] == 0:
            idx[st] = wlo
            continue
        if pol_hi[si] == 1:
            idx[st] = whi
            continue

        lo, hi = wlo, whi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            _, pol, _ = arm.pi_discounted(mid)
            if pol[si] == 1:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol:
                break
        idx[st] = 0.5 * (lo + hi)
    return idx


def compute_whittle_avg(arm, wlo=None, whi=None, tol=1e-4):
    if wlo is None:
        wlo = -(arm.Km + arm.Cm + arm.c_sw + 10.0)
    if whi is None:
        whi = (arm.Km + arm.Cm + arm.c_sw + 10.0)

    idx = {}
    _, pol_lo, _, it_lo = arm.pi_average(wlo)
    _, pol_hi, _, it_hi = arm.pi_average(whi)

    print(f"      Bounds: [{wlo:.1f}, {whi:.1f}]  "
          f"iters: lo={it_lo}, hi={it_hi}  "
          f"active: lo={np.sum(pol_lo)}, hi={np.sum(pol_hi)}")

    for si in range(arm.n):
        st = arm.states[si]
        if pol_lo[si] == 0:
            idx[st] = wlo
            continue
        if pol_hi[si] == 1:
            idx[st] = whi
            continue

        lo, hi = wlo, whi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            _, pol, _, _ = arm.pi_average(mid)
            if pol[si] == 1:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol:
                break
        idx[st] = 0.5 * (lo + hi)
    return idx


# ──────────────────────────────────────────────
#  Parameter Generation
# ──────────────────────────────────────────────

def generate_parameters(M, N, D, alpha, rng):
    machines = []
    for m in range(M):
        raw = rng.uniform(0, 1, size=N)
        raw = np.sort(raw)[::-1]
        p = np.append(raw, 0.0)
        Cm = rng.uniform(D, D + 25)
        Km = alpha * Cm
        machines.append({'p': p, 'Cm': Cm, 'Km': Km})
    return machines


# ──────────────────────────────────────────────
#  Display
# ──────────────────────────────────────────────

def print_machine_params(m_id, params):
    print(f"    Machine {m_id + 1}:")
    print(f"      Cm = {params['Cm']:.4f},  Km = {params['Km']:.4f}")
    p_str = ', '.join([f"{v:.4f}" for v in params['p']])
    print(f"      p  = [{p_str}]")


def print_whittle_table(title, W, N):
    print(f"\n    {title}")
    print(f"    {'x':>3s}  {'W(x,0)[pass]':>14s}  {'W(x,1)[act]':>14s}  {'Diff':>10s}")
    print(f"    {'---':>3s}  {'-'*14:>14s}  {'-'*14:>14s}  {'-'*10:>10s}")
    for x in range(N + 1):
        w0 = W.get((x, 0), float('nan'))
        w1 = W.get((x, 1), float('nan'))
        d  = w1 - w0
        print(f"    {x:>3d}  {w0:>14.6f}  {w1:>14.6f}  {d:>10.6f}")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    M    = 3
    R    = 1
    N    = 7
    beta = 0.95
    c_sw = 3.0

    D     = 10.0
    alpha = 3.0
    NUM_SETS = 5

    print("=" * 70)
    print("  MACHINE REPAIRMAN WITH SWITCHING COST")
    print("  Policy Iteration Based Whittle Index Computation")
    print("=" * 70)
    print(f"  M={M}, R={R}, N={N}, beta={beta}, c_sw={c_sw}")
    print(f"  (D, alpha) = ({D}, {alpha})")
    print(f"  Random parameter sets = {NUM_SETS}")
    print(f"\n  Extended states per machine: {2*(N+1)}")
    print(f"  Passive: {[(x,0) for x in range(N+1)]}")
    print(f"  Active:  {[(x,1) for x in range(N+1)]}")

    # ═══════ CONVERGENCE TEST ═══════
    print(f"\n{'='*70}")
    print("  CONVERGENCE TEST (Policy Iteration)")
    print(f"{'='*70}")

    test_p   = np.array([0.95, 0.85, 0.75, 0.65, 0.5, 0.35, 0.15, 0.0])
    test_arm = MachineArm(N=N, Cm=10.0, Km=30.0, c_sw=c_sw,
                          p=test_p, beta=beta)

    print("\n  Discounted PI:")
    for tw in [-50, -10, 0, 10, 50]:
        t0 = time.time()
        V, pol, it = test_arm.pi_discounted(float(tw))
        dt = time.time() - t0
        print(f"    w={tw:>6.1f}: iters={it:>3d}, time={dt:.4f}s, "
              f"active={np.sum(pol):>2d}")

    print("\n  Average-cost PI:")
    for tw in [-50, -10, 0, 10, 50]:
        t0 = time.time()
        h, pol, g, it = test_arm.pi_average(float(tw))
        dt = time.time() - t0
        print(f"    w={tw:>6.1f}: iters={it:>3d}, time={dt:.4f}s, "
              f"active={np.sum(pol):>2d}, g={g:.4f}")

    # Test Whittle index
    print("\n  Test arm Whittle indices (discounted)...")
    t0 = time.time()
    W_test = compute_whittle_disc(test_arm)
    print(f"    Done in {time.time()-t0:.3f}s")
    print_whittle_table("Test Arm DISCOUNTED", W_test, N)

    print("\n  Test arm Whittle indices (average cost)...")
    t0 = time.time()
    W_test_a = compute_whittle_avg(test_arm)
    print(f"    Done in {time.time()-t0:.3f}s")
    print_whittle_table("Test Arm AVERAGE COST", W_test_a, N)

    # ═══════ MAIN EXPERIMENT ═══════
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

            t0 = time.time()
            W_d = compute_whittle_disc(arms[m])
            t1 = time.time()
            W_a = compute_whittle_avg(arms[m])
            t2 = time.time()
            print(f"    Times: disc={t1-t0:.3f}s, avg={t2-t1:.3f}s")

            print_whittle_table(
                f"Machine {m+1} DISCOUNTED (beta={beta})", W_d, N)
            print_whittle_table(
                f"Machine {m+1} AVERAGE COST", W_a, N)

            disc_indices.append(W_d)
            avg_indices.append(W_a)

        all_disc.append(disc_indices)
        all_avg.append(avg_indices)

        # Policy decisions
        print(f"\n  POLICY DECISIONS (discounted):")
        print(f"  {'Joint State':>45s}  {'W-values':>30s}  {'Assign':>8s}")
        print(f"  {'-'*45}  {'-'*30}  {'-'*8}")

        ex_rng = np.random.default_rng(seed=42 + s)
        for _ in range(6):
            jx  = ex_rng.integers(0, N + 1, size=M)
            jap = ex_rng.integers(0, 2, size=M)
            wv  = [disc_indices[m][(jx[m], jap[m])] for m in range(M)]
            best = np.argmax(wv)
            idle = all(w < 0 for w in wv)
            st_str = "  ".join(
                [f"M{m+1}=({jx[m]},{jap[m]})" for m in range(M)])
            wv_str = "  ".join([f"{w:.2f}" for w in wv])
            dec = "IDLE" if idle else f"M{best+1}"
            print(f"  {st_str:>45s}  {wv_str:>30s}  {dec:>8s}")

    # ═══════ SUMMARY ═══════
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
            for si in range(NUM_SETS):
                for m in range(M):
                    vals.append(all_disc[si][m][(x, ap)])
            vals = np.array(vals)
            fam = 'P' if ap == 0 else 'A'
            print(f"  {f'({x},{ap})[{fam}]':>10s}  {vals.min():>10.4f}  "
                  f"{np.percentile(vals,25):>10.4f}  "
                  f"{np.median(vals):>10.4f}  "
                  f"{np.percentile(vals,75):>10.4f}  "
                  f"{vals.max():>10.4f}")

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
            for si in range(NUM_SETS):
                for m in range(M):
                    vals.append(all_avg[si][m][(x, ap)])
            vals = np.array(vals)
            fam = 'P' if ap == 0 else 'A'
            print(f"  {f'({x},{ap})[{fam}]':>10s}  {vals.min():>10.4f}  "
                  f"{np.percentile(vals,25):>10.4f}  "
                  f"{np.median(vals):>10.4f}  "
                  f"{np.percentile(vals,75):>10.4f}  "
                  f"{vals.max():>10.4f}")

    # Switching cost effect
    print(f"\n{'='*70}")
    print(f"  SWITCHING COST EFFECT: W(x,1) - W(x,0)  (c_sw={c_sw})")
    print(f"{'='*70}")
    print(f"  {'x':>3s}  {'Disc mean':>12s}  {'Disc std':>10s}  "
          f"{'Avg mean':>12s}  {'Avg std':>10s}")
    print(f"  {'---':>3s}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*10}")

    for x in range(N + 1):
        dd, da = [], []
        for si in range(NUM_SETS):
            for m in range(M):
                dd.append(all_disc[si][m][(x,1)] - all_disc[si][m][(x,0)])
                da.append(all_avg[si][m][(x,1)]  - all_avg[si][m][(x,0)])
        dd, da = np.array(dd), np.array(da)
        print(f"  {x:>3d}  {dd.mean():>12.6f}  {dd.std():>10.6f}  "
              f"{da.mean():>12.6f}  {da.std():>10.6f}")

    print(f"\n{'='*70}")
    print("  DONE.")
    print(f"{'='*70}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Total runtime: {time.time()-t0:.2f}s")