import numpy as np

def compute_G_H(p, beta):
    """ Computes G(k) and H(k) based on transition probabilities. """
    num_states = len(p)
    G = np.zeros(num_states + 1)
    H = np.zeros(num_states)
    
    for k in range(num_states):
        H[k] = beta**(k+1) * np.prod(p[:k])
        G[k+1] = G[k] + (1 - p[k]) * H[k]
    
    return G, H

def whittle_index(K, C, p, beta=0.95):
    """ Computes the Whittle index for a machine. """
    G, H = compute_G_H(p, beta)
    num_states = len(p)
    W = np.zeros(num_states)

    for k in range(num_states):
        numerator = (1 - G[k+1]) - p[k] * (1 - beta * G[k])
        denominator = (1 - G[k+1]) - beta * p[k] * (1 - G[k])
        W[k] = K * (numerator / denominator) - C

    return W





# Example usage
num_damage_states = 8
p = np.sort(np.random.uniform(0.1, 0.9, num_damage_states))[::-1]  # Decreasing failure probabilities
K = 100  # Cost of catastrophic failure
C = 10   # Repair cost

whittle_indices = whittle_index(K, C, p)
print("Whittle Indices:", whittle_indices)


# Constants
NUM_MACHINES = 3
NUM_DAMAGE_STATES = 8
NUM_EPISODES = 100
NUM_PROBLEMS = 5120
DISCOUNT_FACTOR = 0.95

def whittle_paper_discounted(K, C, p, beta=0.95):
    """ Computes the Whittle index for a machine. """
    G, H = compute_G_H(p, beta)
    num_states = len(p)
    W = np.zeros(num_states)

    for k in range(num_states):
        print(f" {k}th Iteration")
        numerator = (1 - G[k+1]) - p[k] * (1 - beta * G[k])
        denominator = (1 - G[k+1]) - beta * p[k] * (1 - G[k])
        W[k] = K * (numerator / denominator) - C

        print("W[k] : ", W[k])
    print("Length of W : ", len(W))
    return W

whittle_indices = whittle_paper_discounted(K, C, p)
print("Whittle Paper Indices:", whittle_indices)