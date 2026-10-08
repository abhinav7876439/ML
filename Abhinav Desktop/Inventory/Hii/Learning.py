import numpy as np
import random
import math
import matplotlib.pyplot as plt

# ============================================================
# Whittle index helper class
# ============================================================

class MakeToStockWhittle:
    def __init__(self, lam, mu, h, D, max_state_cache=1000):
        self.lam = float(lam)
        self.mu = float(mu)
        self.h = float(h)
        self.D = float(D)
        self.s = self.D * self.lam
        self.p = self.lam / self.mu if self.mu > 0 else 1.0
        self._cache = {}
        self.max_cache = max_state_cache

    # -------- Closed-form Whittle index --------
    def whittle_index_B(self, B):
        p = self.p
        s = self.s
        h = self.h
        if abs(1.0 - p) < 1e-12:
            return -s / max(p, 1e-12) + h * (B * (B + 1)) / 2.0
        pB = p ** B
        num = 1.0 - (B + 1) * pB + B * (p ** (B + 1))
        den = ((1.0 - p) ** 2) * pB
        return -s / p + h * (num / den)

    def whittle_index(self, x):
        x = int(x)
        if x <= self.max_cache and x in self._cache:
            return self._cache[x]
        B = x + 1
        w = self.whittle_index_B(B)
        if x <= self.max_cache:
            self._cache[x] = w
        return w

    # -------- Numeric Whittle index via subsidy search --------
    def _q_diff(self, x, subsidy, Bmax=20, beta=0.95):
        """ Difference in value between passive and active if subsidy is applied. """
        lam, mu, h, D = self.lam, self.mu, self.h, self.D
        # Passive action: pay holding + lost sale penalty ± subsidy
        cost_passive = h * x + (D if x==0 else 0) * lam - subsidy
        # Active: holding cost but production replenishes
        cost_active = h * x
        return cost_passive - cost_active

    def numeric_whittle_index(self, x, Bmax=20, tol=1e-4):
        """ Binary search to find subsidy w where passive=active. """
        low, high = -100.0, 100.0
        for _ in range(60):
            mid = 0.5*(low+high)
            diff = self._q_diff(x, subsidy=mid, Bmax=Bmax)
            if diff > 0:
                low = mid
            else:
                high = mid
        return 0.5*(low+high)


# ============================================================
# Multi-class event simulator
# ============================================================

class MultiClassSimulator:
    def __init__(self, classes):
        self.K = len(classes)
        self.classes = classes
        self.whittles = [MakeToStockWhittle(c['lambda'], c['mu'], c['h'], c['D']) for c in classes]

    def _choose_whittle(self, x):
        idxs = [self.whittles[k].whittle_index(x[k]) for k in range(self.K)]
        maxI = max(idxs)
        candidates = [i for i, val in enumerate(idxs) if abs(val - maxI) < 1e-12]
        return random.choice(candidates)

    def _choose_myopic(self, x):
        scores = []
        for k in range(self.K):
            lam = self.classes[k]['lambda']
            D = self.classes[k]['D']
            h = self.classes[k]['h']
            score = (D*lam if x[k] == 0 else 0.0) - h
            scores.append(score)
        maxS = max(scores)
        cand = [i for i, s in enumerate(scores) if abs(s - maxS) < 1e-12]
        return random.choice(cand)

    def _choose_highest_demand(self, x):
        lambdas = [c['lambda'] for c in self.classes]
        maxL = max(lambdas)
        cand = [i for i, l in enumerate(lambdas) if abs(l - maxL) < 1e-12]
        return random.choice(cand)

    def _choose_highest_stockout(self, x):
        scores = [c['lambda'] * (1 if x[k] == 0 else 0) for k, c in enumerate(self.classes)]
        maxS = max(scores)
        cand = [i for i, s in enumerate(scores) if abs(s - maxS) < 1e-12]
        return random.choice(cand)

    def _choose_round_robin(self, x, last_rr):
        return (last_rr + 1) % self.K

    def simulate(self, T=1000.0, seed=None, policy='whittle', verbose=False):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        K = self.K
        params = self.classes
        t = 0.0
        x = [int(p.get('init_x',0)) for p in params]

        next_demand = [(np.random.exponential(1.0/p['lambda']) if p['lambda']>0 else float('inf')) for p in params]

        last_rr = -1
        if policy == 'whittle':
            server_class = self._choose_whittle(x)
        elif policy == 'myopic':
            server_class = self._choose_myopic(x)
        elif policy == 'highest_demand':
            server_class = self._choose_highest_demand(x)
        elif policy == 'highest_stockout':
            server_class = self._choose_highest_stockout(x)
        elif policy == 'round_robin':
            server_class = 0
            last_rr = 0
        else:
            raise ValueError("Unknown policy")

        next_service = t + np.random.exponential(1.0/params[server_class]['mu'])

        total_cost = 0.0
        total_time = 0.0
        lost_sales_count = [0]*K
        cumulative_holding_time = [0.0]*K

        times = [0.0]
        trajectories = [[xi] for xi in x]

        while t < T:
            nd_idx = int(np.argmin(next_demand))
            nd_time = next_demand[nd_idx]
            next_event_time = min(nd_time, next_service)
            if next_event_time == float('inf'):
                break

            dt = next_event_time - t
            if dt < -1e-12: break

            inst_cost_rate = 0.0
            for k in range(K):
                hk = params[k]['h']
                inst_cost_rate += hk * x[k]
                cumulative_holding_time[k] += x[k]*dt
            total_cost += inst_cost_rate*dt
            total_time += dt
            t = next_event_time

            if abs(next_service - t) < 1e-12:
                x[server_class] += 1
                if policy == 'whittle':
                    server_class = self._choose_whittle(x)
                elif policy == 'myopic':
                    server_class = self._choose_myopic(x)
                elif policy == 'highest_demand':
                    server_class = self._choose_highest_demand(x)
                elif policy == 'highest_stockout':
                    server_class = self._choose_highest_stockout(x)
                elif policy == 'round_robin':
                    last_rr = (last_rr+1)%K
                    server_class = last_rr
                next_service = t + np.random.exponential(1.0/params[server_class]['mu'])

            if abs(nd_time - t) < 1e-12:
                k = nd_idx
                if x[k] > 0:
                    x[k] -= 1
                else:
                    lost_sales_count[k] += 1
                    total_cost += params[k]['D']
                lamk = params[k]['lambda']
                next_demand[k] = t + (np.random.exponential(1.0/lamk) if lamk>0 else float('inf'))

            times.append(t)
            for k in range(K): trajectories[k].append(x[k])

        avg_cost = total_cost / total_time if total_time>0 else float('inf')
        return {
            'times': times,
            'trajectories': trajectories,
            'avg_cost': avg_cost,
            'total_cost': total_cost,
            'total_time': total_time,
            'lost_sales_count': lost_sales_count,
            'cumulative_holding_time': cumulative_holding_time
        }

    def batch_compare_policies(self, T=2000.0, n_rep=20, seed=123, 
                               policies=('whittle','myopic','round_robin','highest_demand','highest_stockout')):
        results = {}
        rng = random.Random(seed)
        for pol in policies:
            costs = []
            for rep in range(n_rep):
                s = rng.randint(0,2**31-1)
                res = self.simulate(T=T, seed=s, policy=pol)
                costs.append(res['avg_cost'])
            arr = np.array(costs)
            mean = arr.mean()
            stderr = arr.std(ddof=1)/math.sqrt(len(arr))
            ci95 = (mean-1.96*stderr, mean+1.96*stderr)
            results[pol] = {'costs':costs, 'mean':mean, 'ci95':ci95}
        return results


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    classes = [
        {'lambda': 0.8, 'mu': 1.2, 'h': 0.5, 'D': 10.0, 'init_x': 1},
        {'lambda': 0.6, 'mu': 1.0, 'h': 0.4, 'D': 8.0,  'init_x': 0},
        {'lambda': 1.0, 'mu': 1.5, 'h': 0.7, 'D': 12.0, 'init_x': 2}
    ]
    sim = MultiClassSimulator(classes)

    # --- Compare trajectories under Whittle
    res_w = sim.simulate(T=400.0, seed=42, policy='whittle')
    times, trajs = res_w['times'], res_w['trajectories']
    plt.figure(figsize=(10,5))
    for k in range(len(classes)):
        plt.step(times, trajs[k], where='post', label=f'Class {k+1}')
    plt.xlabel("Time"); plt.ylabel("Inventory")
    plt.title("Inventory trajectories under Whittle (single run)")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    print("Whittle single-run avg cost:", res_w['avg_cost'])
    print("Whittle single-run lost sales:", res_w['lost_sales_count'])

    # --- Compare policies
    policies = ('whittle','myopic','round_robin','highest_demand','highest_stockout')
    batch_res = sim.batch_compare_policies(T=2000.0, n_rep=30, seed=2025, policies=policies)
    print("\nPolicy comparison (avg cost per unit time):")
    for pol in policies:
        r = batch_res[pol]
        print(f"{pol:16s} mean={r['mean']:.4f}  95% CI=({r['ci95'][0]:.4f},{r['ci95']:.4f})")

    means = [batch_res[p]['mean'] for p in policies]
    cis = [batch_res[p]['ci95'] for p in policies]
    lower_err = [means[i]-cis[i] for i in range(len(policies))]
    upper_err = [cis[i]-means[i] for i in range(len(policies))]
    err = np.array([lower_err, upper_err])
    plt.figure(figsize=(7,4))
    plt.bar(policies, means, yerr=err, capsize=8)
    plt.ylabel("Avg cost / unit time")
    plt.title("Policy comparison (mean ±95% CI)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(); plt.show()

    # --- Closed-form vs Numeric Whittle index
    w = MakeToStockWhittle(0.8, 1.2, 0.5, 10.0)
    states = range(6)
    print("\nClosed form vs Numeric Whittle indices:")
    for s in states:
        cf = w.whittle_index(s)
        num = w.numeric_whittle_index(s, Bmax=20)
        print(f"State {s}: closed-form={cf:.4f}, numeric≈{num:.4f}")

    # --- Lost sales per class for Whittle
    plt.figure(figsize=(6,4))
    plt.bar([f"C{k+1}" for k in range(len(classes))], res_w['lost_sales_count'])
    plt.xlabel("Class"); plt.ylabel("Lost sales (count)")
    plt.title("Lost sales per class (Whittle, single run)")
    plt.tight_layout(); plt.show()
