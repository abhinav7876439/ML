import numpy as np
import random, math
import matplotlib.pyplot as plt

# =====================================================
# Closed-form Whittle index helper
# =====================================================
class MakeToStockWhittle:
    def __init__(self, lam, mu, h, D, max_state_cache=1000):
        self.lam = float(lam)
        self.mu = float(mu)
        self.h = float(h)
        self.D = float(D)
        self.s = self.D * self.lam      # lost-sale cost rate
        self.p = self.lam / self.mu if self.mu > 0 else 1.0
        self._cache = {}
        self.max_cache = max_state_cache

    def whittle_index_B(self, B):
        p, s, h = self.p, self.s, self.h
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


# =====================================================
# Numeric Whittle index computation via value iteration
# =====================================================
def solve_single_arm_policy(lam, mu, h, D, W, N, tol=1e-6, max_iter=5000):
    """ Relative value iteration for one arm truncated at N. """
    q = lam + mu
    if q <= 0: q = 1.0

    def cost(x, a):
        base = (h * x) / q
        if x == 0: base += (D * lam) / q
        base += (W * (1 - a)) / q
        return base

    def trans_probs(x, a):
        p = np.zeros(N+1)
        if a == 1:
            if x < N: p[x+1] += mu/q
            else: p[N] += mu/q
        if x > 0: p[x-1] += lam/q
        p[x] += 1.0 - p.sum()
        return p

    v = np.zeros(N+1)
    for _ in range(max_iter):
        v_new = np.zeros_like(v)
        for x in range(N+1):
            vals = []
            for a in (0,1):
                vals.append(cost(x,a) + trans_probs(x,a).dot(v))
            v_new[x] = min(vals)
        shift = v_new[0]; v_new -= shift
        if np.max(np.abs(v_new-v))<tol: break
        v = v_new
    policy = []
    for x in range(N+1):
        vals = [cost(x,a) + trans_probs(x,a).dot(v) for a in (0,1)]
        policy.append(int(np.argmin(vals)))
    return policy

def whittle_index_numeric(lam, mu, h, D, x, N=30,
                          W_low=-1e4, W_high=1e4, tol=1e-2):
    def passive(W): return solve_single_arm_policy(lam,mu,h,D,W,N)[x]==0
    if passive(W_low): return W_low
    if not passive(W_high): return W_high
    a,b = W_low,W_high
    for _ in range(50):
        m=0.5*(a+b)
        if passive(m): b=m
        else: a=m
        if abs(b-a)<tol: break
    return 0.5*(a+b)


# =====================================================
# Multi-class simulator
# =====================================================
class MultiClassSimulator:
    def __init__(self, classes):
        self.K = len(classes)
        self.classes = classes
        self.whittles = [MakeToStockWhittle(c['lambda'],c['mu'],c['h'],c['D']) for c in classes]

    def _choose_whittle(self,x):
        idxs=[self.whittles[k].whittle_index(x[k]) for k in range(self.K)]
        m=max(idxs)
        cand=[i for i,v in enumerate(idxs) if abs(v-m)<1e-12]
        return random.choice(cand)

    def _choose_myopic(self,x):
        scores=[]
        for k in range(self.K):
            lam,D,h=self.classes[k]['lambda'],self.classes[k]['D'],self.classes[k]['h']
            scores.append(D*lam if x[k]==0 else -h)
        m=max(scores)
        cand=[i for i,v in enumerate(scores) if abs(v-m)<1e-12]
        return random.choice(cand)

    def _choose_round_robin(self,last): return (last+1)%self.K

    def _choose_highest_demand(self,x):
        # new baseline: always serve class with largest λ
        lam_list=[c['lambda'] for c in self.classes]
        m=max(lam_list)
        cand=[i for i,l in enumerate(lam_list) if abs(l-m)<1e-12]
        return random.choice(cand)

    def simulate(self,T=1000.0,seed=None,policy='whittle'):
        if seed is not None: np.random.seed(seed); random.seed(seed)
        t=0.0; x=[int(c.get('init_x',0)) for c in self.classes]
        next_demand=[np.random.exponential(1/c['lambda']) for c in self.classes]
        if policy=='whittle': server=self._choose_whittle(x)
        elif policy=='myopic': server=self._choose_myopic(x)
        elif policy=='round_robin': server=0; last=0
        elif policy=='highest_demand': server=self._choose_highest_demand(x)
        else: raise ValueError
        next_service=t+np.random.exponential(1/self.classes[server]['mu'])
        total_cost=0.0; total_time=0.0
        lost_sales=[0]*self.K
        times=[0.0]; trajs=[[xi] for xi in x]
        while t<T:
            nd_idx=int(np.argmin(next_demand))
            nd_time=next_demand[nd_idx]
            ev_time=min(nd_time,next_service)
            dt=ev_time-t; 
            if dt<0: break
            # accumulate holding cost
            total_cost+=sum(self.classes[k]['h']*x[k] for k in range(self.K))*dt
            total_time+=dt; t=ev_time
            if abs(next_service-t)<1e-12:
                x[server]+=1
                if policy=='whittle': server=self._choose_whittle(x)
                elif policy=='myopic': server=self._choose_myopic(x)
                elif policy=='round_robin': last=(server+1)%self.K; server=last
                elif policy=='highest_demand': server=self._choose_highest_demand(x)
                next_service=t+np.random.exponential(1/self.classes[server]['mu'])
            if abs(nd_time-t)<1e-12:
                if x[nd_idx]>0: x[nd_idx]-=1
                else: lost_sales[nd_idx]+=1; total_cost+=self.classes[nd_idx]['D']
                next_demand[nd_idx]=t+np.random.exponential(1/self.classes[nd_idx]['lambda'])
            times.append(t)
            for k in range(self.K): trajs[k].append(x[k])
        avg_cost=total_cost/total_time
        return {'times':times,'trajectories':trajs,'avg_cost':avg_cost,'lost_sales':lost_sales}


# =====================================================
# Run demo
# =====================================================
if __name__=="__main__":
    classes=[{'lambda':0.8,'mu':1.2,'h':0.5,'D':10.0,'init_x':1},
             {'lambda':0.6,'mu':1.0,'h':0.4,'D':8.0,'init_x':0},
             {'lambda':1.0,'mu':1.5,'h':0.7,'D':12.0,'init_x':2}]
    sim=MultiClassSimulator(classes)

    # Single-run trajectory
    res=sim.simulate(T=300,seed=42,policy='whittle')
    for k in range(len(classes)):
        plt.step(res['times'],res['trajectories'][k],where='post',label=f"Class{k+1}")
    plt.legend(); plt.title("Whittle trajectories"); plt.show()

    print("Whittle avg cost:",res['avg_cost'])
    print("Lost sales:",res['lost_sales'])

    # Batch compare
    policies=['whittle','myopic','round_robin','highest_demand']
    results={}
    for pol in policies:
        costs=[sim.simulate(T=1000,seed=i,policy=pol)['avg_cost'] for i in range(20)]
        arr=np.array(costs); mean=arr.mean(); stderr=arr.std(ddof=1)/math.sqrt(len(arr))
        ci=(mean-1.96*stderr,mean+1.96*stderr)
        results[pol]={'mean':mean,'ci':ci}
    for pol in policies:
        r=results[pol]
        print(f"{pol:15s}: mean={r['mean']:.3f}, 95%CI={r['ci']}")

    # Closed-form vs numeric Whittle indices
    cls=classes[0]
    closed=[sim.whittles[0].whittle_index(x) for x in range(15)]
    numeric=[whittle_index_numeric(cls['lambda'],cls['mu'],cls['h'],cls['D'],x,N=20) for x in range(15)]
    plt.plot(closed,label="Closed-form")
    plt.plot(numeric,label="Numeric (VI)")
    plt.xlabel("Inventory level x"); plt.ylabel("Index")
    plt.title("Whittle indices: closed vs numeric (class1)")
    plt.legend(); plt.show()
