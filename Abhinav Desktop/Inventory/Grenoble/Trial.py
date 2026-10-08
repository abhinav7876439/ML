import numpy as np

def next_state(x, a, mu, lam, gamma, max_inv, dt=1.0):
    """
    Simulates one-step transition in integer-valued inventory dynamics.

    Parameters:
        x        : current inventory level (integer)
        a        : action (0 = passive, 1 = active)
        mu       : exponential rate of production (if active)
        lam      : Poisson demand rate
        gamma    : perishability rate (per item per unit time) exponentially distributed
        max_inv  : maximum inventory (capacity)
        dt       : time step

    Returns:
        x_next   : next inventory level (integer)
    """
    # --- Production ---
    if a == 1:
        produce = np.random.exponential(1/mu * dt)
        print(f"Produced (raw): {produce:.4f}")
        produced = int(produce)  # convert to integer, mean=1/mu
    else:
        produced = 0

    # --- Demand arrivals ---
    demand = np.random.poisson(lam * dt)
    print(f"demand : {demand:.4f}")
    
    # --- Perishability ---
    prob = np.exp(-gamma * dt)
    print(f"survive prob: {prob:.4f}")
    perish_prob = 1 - prob
    print(f"perish prob: {perish_prob:.4f}")
    perished = np.random.binomial(x, perish_prob)
    print(f"perished: {perished:.4f}")

    # --- Inventory update ---
    x_next = x + produced - demand - perished
    print(f"Next inventory (raw): {x_next:.4f}")
    x_next = max(0, min(int(round(x_next)), max_inv))  # enforce boundaries and integer

    return x_next


# Parameters
mu = 0.8         # production rate
lam = 0.6        # demand rate
gamma = 0.1      # perishability rate
max_inv = 10     # max inventory

x = 0  # initial inventory

for t in range(13):
    a = np.random.choice([0, 1])  # randomly pick action
    x = next_state(x, a, mu, lam, gamma, max_inv)
    print(f"t={t}, a={a}, inventory={x}")



########### Trial: Two time-scale stochastic approximation for Poisson rate estimation ###############################################


import numpy as np

# Simulate inter-arrival times (exponential with true lambda)
true_lambda = 0.5
N = 200
inter_arrival_times = np.random.exponential(1/true_lambda, N)

# Two time-scale learning rates
alpha = 0.1   # fast time-scale (mean)
beta = 0.01   # slow time-scale (rate)

# Initial estimates
mean_hat = 1.0
lambda_hat = 1.0

for delta in inter_arrival_times:
    # Fast time-scale: update mean estimate
    mean_hat += alpha * (delta - mean_hat)
    # Slow time-scale: update lambda estimate
    lambda_hat += beta * ((1/mean_hat) - lambda_hat)

print(f"True lambda: {true_lambda:.3f}")
print(f"Estimated lambda (two time-scale): {lambda_hat:.3f}")