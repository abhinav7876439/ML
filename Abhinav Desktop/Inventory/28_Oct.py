import numpy as np

# ------------------------------
# Parameters
# ------------------------------
I_max = 50        # Maximum inventory
h = 0.01          # Holding cost per unit
p = 0.02          # Perish/decay cost per unit
lam = 0.02        # Lost sales cost if inventory is 0
mu = 0.2          # Production rate
gamma = 0.05      # Demand/decay rate
tau0 = 1.0        # Duration of action 0
tau1 = 1.0        # Duration of action 1
C_switch = 0.05   # Switching cost for 0->1
w_scale = 0.03    # Reward for producing

# ------------------------------
# Base cost and immediate cost function
# ------------------------------
def C_base(i):
    return h*i + p*gamma*i + lam*(i==0)

def c(i, a_prev, a):
    cost = C_base(i)
    if a == 1:
        cost -= w_scale
    if a_prev == 0 and a == 1:
        cost += C_switch
    return cost

# ------------------------------
# Transition probabilities
# ------------------------------
def transition_probs(i, a):
    """Returns dict {i_next: probability}"""
    if a == 0:
        r_up = 0
    else:
        r_up = mu
    r_down = i*gamma + lam if i > 0 else 0
    R = r_up + r_down
    
    if R == 0:
        return {i: 1.0}
    
    p_any = 1 - np.exp(-R*(tau1 if a==1 else tau0))
    p_up = r_up/R * p_any if r_up>0 else 0
    p_down = r_down/R * p_any if r_down>0 else 0
    p_stay = 1 - p_any
    
    probs = {}
    # Up
    if i+1 <= I_max:
        probs[i+1] = p_up
    else:
        probs[i] = p_up  # cap at I_max
    
    # Down
    if i-1 >= 0:
        probs[i-1] = p_down
    else:
        probs[i] = probs.get(i,0) + p_down
    
    # Stay
    probs[i] = probs.get(i,0) + p_stay
    
    return probs

# ------------------------------
# Relative Value Iteration (RVI)
# ------------------------------
def RVI(max_iter=10000, tol=1e-6, ref_state=(0,0)):
    V = np.zeros((I_max+1,2))  # V[i, a_prev]
    rho = 0.0
    
    for it in range(max_iter):
        V_new = np.zeros_like(V)
        for i in range(I_max+1):
            for a_prev in [0,1]:
                Qs = []
                for a in [0,1]:
                    cost = c(i,a_prev,a)
                    probs = transition_probs(i,a)
                    value = sum(probs[i_next]*V[i_next,a] for i_next in probs)
                    Qs.append(cost + value)
                V_new[i,a_prev] = min(Qs)
        # RVI adjustment
        diff = V_new - V_new[ref_state]
        V_new -= V_new[ref_state]
        rho_new = V_new[ref_state]
        
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            rho = rho_new
            break
        V = V_new
        rho = rho_new
    
    # Optimal policy
    pi = np.zeros((I_max+1,2),dtype=int)
    for i in range(I_max+1):
        for a_prev in [0,1]:
            Qs = []
            for a in [0,1]:
                cost = c(i,a_prev,a)
                probs = transition_probs(i,a)
                value = sum(probs[i_next]*V[i_next,a] for i_next in probs)
                Qs.append(cost + value)
            pi[i,a_prev] = np.argmin(Qs)
    
    return rho, V, pi

# ------------------------------
# Run RVI
# ------------------------------
rho, V, pi = RVI()
print(f"Average Cost (rho): {rho:.4f}")
print("Optimal Policy (i, a_prev) -> a_t:")
for i in range(I_max+1):
    for a_prev in [0,1]:
        print(f"({i},{a_prev}) -> {pi[i,a_prev]}")

