
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import bisect
import networkx as nx
from scipy.optimize import brentq
# Try to import markovianbandit; if not available, skip package comparison
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; skip package comparison.")

def build_extended_mdp(N, p, C, K, C_switch):
    num_states = N + 2
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)

    # State labels
    state_labels = ["(0,1)", "(0,0)"] + [f"({i},0)" for i in range(1, N+1)]

    # Active transitions
    for s in range(num_states):
        if s == 0:
            P1[s, 0] = 1.0
            R1[s] = -C
        else:
            P1[s, 0] = 1.0
            R1[s] = -(C + C_switch)

    # Passive transitions
    for s in range(num_states):
        if s == 0:
            P0[s, 2] = p[0]
            P0[s, 1] = 1-p[0]
            R0[s] = -(K*(1-p[0]))
        elif s == 1:
            P0[s, 2] = p[0]
            P0[s, 1] = 1-p[0]
            R0[s] = -(K*(1-p[0]))
        else:
            idx = s-1
            if idx < N:
                P0[s, s+1] = p[idx]
                P0[s, 1] = 1-p[idx]
                R0[s] = -(K*(1-p[idx]))
            else:
                P0[s, 1] = 1.0
                R0[s] = -K

    return P0, P1, R0, R1, state_labels


def print_transition_matrix(P, state_labels, action_name):
    print("="*70)
    print(f"Transition matrix for action {action_name}")
    print("="*70)
    df = pd.DataFrame(P, index=state_labels, columns=state_labels)
    print(df)


def print_reward_vector(R, state_labels, action_name):
    print("="*70)
    print(f"Reward vector for action {action_name}")
    print("="*70)
    df = pd.DataFrame(R, index=state_labels, columns=[action_name])
    print(df)






# ------------------------------------------------------------
#  Closed-form expressions
# ------------------------------------------------------------
# def compute_G_H(p, beta):
#     """ Computes G(k) and H(k) as defined in the paper. """
#     N = len(p)
#     H = np.zeros(N)
#     G = np.zeros(N+1)
#     for k in range(N):
#         prod = 1.0
#         for i in range(k):
#             prod *= p[i]
#         H[k] = beta**(k+1) * prod
#         G[k+1] = G[k] + (1 - p[k]) * H[k]
#     return G, H

# def closed_form_indices_discounted(p, C, K, beta, c_switch):
#     """
#     Discounted Whittle indices for states x = 0..N-1 (active states).
#     p[0..N-1] are the survival probabilities; p[N] is not used.
#     """
#     N = len(p)          # p has length N (original N, without terminal zero)
#     G, H = compute_G_H(p, beta)
#     W = np.zeros(N)
#     for k in range(N):
#         num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
#         den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
#         W[k] = K * (num / den) - C - c_switch  # c_switch * (k == 0)
#     return W



def compute_G_H(p, beta):
    """
    Compute the G and H sequences defined in the paper.
    
    Parameters:
    p : array of length N, survival probabilities p[0]..p[N-1]
    beta : discount factor (0 < beta < 1)
    
    Returns:
    G : array of length N+1, G[0]=0, G[k+1] = G[k] + (1-p[k]) * H[k]
    H : array of length N, H[k] = beta^(k+1) * prod_{i=0}^{k-1} p[i]
    """
    N = len(p)
    H = np.zeros(N)
    G = np.zeros(N+1)          # G[0] = 0
    prod = 1.0
    for k in range(N):
        # H[k] = beta^(k+1) * prod_{i=0}^{k-1} p[i]
        H[k] = beta**(k+1) * prod
        G[k+1] = G[k] + (1 - p[k]) * H[k]
        prod *= p[k]           # update product for next iteration
    return G, H

def closed_form_indices(p, beta, K, C, C_switch):
    """
    Compute closed-form Whittle indices for passive states (k,0) and the
    special active state (0,1).
    
    Returns:
    W_passive : array of length N, indices for k = 0..N-1
    W_active_special : float, index for state (0,1)
    """
    N = len(p)
    G, H = compute_G_H(p, beta)
    
    # Passive states (k,0)
    W_passive = np.zeros(N)
    for k in range(N):
        # Using formulas from the paper
        num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
        den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
        # Avoid division by zero (should not happen for valid parameters)
        W_passive[k] = K * (num / den) - C - C_switch
    
    # Special active state (0,1) – using k = 0 in the active-state formula
    # i.e., G[1] and H[1] (note H[1] exists only if N >= 2; for N=1 we treat H[1]=0)
    if N >= 2:
        G1 = G[1]
        H1 = H[1]
    else:
        # For N=1 there is only p[0]; H[1] would be beta^2 * p[0] but not defined.
        # In the paper, the sum for the active state uses H(2) if needed.
        # For simplicity we assume N>=2; otherwise adjust accordingly.
        # Here we set H1 = 0 to avoid errors, but the formula may not be accurate.
        G1 = G[1]
        H1 = 0.0
    
    numerator = (1 - beta) * (K * G1 + C_switch * H1)
    denominator = beta * (1 - G1) - H1
    W_active = (numerator / denominator) - C if denominator != 0 else np.inf
    
    return W_passive, W_active






def build_mdp(p, C, K):
    """
    Builds the complete MDP (transition and reward matrices) using provided survival probabilities p.
    
    Parameters:
    p: array of survival probabilities (decreasing order) of length N
    C: repair cost
    K: penalty cost for reset to pristine
    
    Returns:
    P0: transition matrix for passive mode (no repair) of shape (N+1, N+1) or (N, N)
    P1: transition matrix for repair mode of shape (N+1, N+1) or (N, N)
    R0: reward vector for passive mode
    R1: reward vector for repair mode
    """
    N = len(p) + 1  # Number of non-terminal states
    
    # Build transition matrices
    P0 = np.zeros((N, N))
    P1 = np.zeros((N, N))
    
    # Fill P0 (No repair scenario) using p
    for i in range(N - 1):
        #P0[i, i + 1] = 1 - p[i]  # Move to worse state
        P0[i, i + 1] =  p[i]  # Move to worse state
        P0[i, 0] = 1 - p[i]           # Reset to pristine
    
    P0[N - 1, 0] = 1  # Most damaged state resets to pristine
    
    # Fill P1 (With repair scenario)
    P1[:, 0] = 1  # Repair resets any state to pristine
    
    # Build reward matrices
    R0 = np.zeros(N)
    R1 = np.zeros(N)
    
    # Passive rewards: penalty when transitioning to pristine
    for k in range(N):
        R0[k] = -P0[k, 0] * K
    
    # Repair rewards: fixed cost
    R1[:] = -C
    
    # Define state labels (just the damage states since this is a single machine)
    state_labels = [f"State {i} (damage level {i})" for i in range(N)]
    
    return P0, P1, R0, R1, state_labels




# ============================================================================
# Example usage and comparison with the markovianbandit package
# ============================================================================
if __name__ == "__main__":
    # Parameters
    N = 4
    np.random.seed(42)
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]   # decreasing survival probabilities
    p_full = np.append(p, 0.0)                       # terminal zero for the extended MDP
    C, K = 5.0, 500.0
    C_switch = 1.0
    beta = 0.95

        
    # Build MDP without terminal state
    P0, P1, R0, R1, state_labels = build_mdp(p, C, K)
    # Print matrices with state labels both vertically and horizontally
    print_transition_matrix(P0, state_labels, "Passive (a=0)")
    print_reward_vector(R0, state_labels, "Passive (a=0)")

    print_transition_matrix(P1, state_labels, "Active (a=1)")
    print_reward_vector(R1, state_labels, "Active (a=1)")

    
    print("=" * 70)
    print(f"Survival probabilities p[0..{N}]: {np.round(p_full, 3)}")
    print("=" * 70)

    P0_ext, P1_ext, R0_ext, R1_ext, state_labels_ext = build_extended_mdp(N, p_full, C, K, C_switch)

    # Print matrices with state labels both vertically and horizontally
    print_transition_matrix(P0_ext, state_labels_ext, "Passive (a=0)")
    print_reward_vector(R0_ext, state_labels_ext, "Passive (a=0)")

    print_transition_matrix(P1_ext, state_labels_ext, "Active (a=1)")
    print_reward_vector(R1_ext, state_labels_ext, "Active (a=1)")
    
    # Closed‑form indices
    W_passive, W_active = closed_form_indices(p_full, beta, K, C, C_switch)
    
    print("\nClosed‑form Whittle indices (discounted, beta = {}):".format(beta))
    print(f"  Active state (0,1): W = {W_active:.6f}")
    for k in range(N):
        print(f"  Passive state ({k},0): W = {W_passive[k]:.6f}")
    
    # Compare with markovianbandit package if available
    if PKG_AVAILABLE:
        # Build the extended MDP using the full p (including terminal zero)
        P0, P1, R0, R1, _ = build_extended_mdp(N, p_full, C, K, C_switch)
        
        # Create restless bandit model
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
        
        # Discounted indices from the package
        pkg_disc = model.whittle_indices(discount=beta)
        pkg_passive = pkg_disc[1:N+2]          # states (0,0) ... (N-1,0)
        pkg_active_special = pkg_disc[0]       # state (0,1)


        print("\nPackage discounted indices (beta = {}):".format(beta))
        print(f"  Active state (0,1): W = {pkg_active_special:.6f}")
        for k in range(N):
            print(f"  Passive state ({k},0): W = {pkg_passive[k]:.6f}")
        
        # Differences
        print("\nDifferences (closed‑form - package):")
        print(f"  Active (0,1): {W_active - pkg_active_special:.2e}")
        for k in range(N):
            diff = W_passive[k] - pkg_passive[k]
            print(f"  Passive ({k},0): {diff:.2e}")


        # Build the simple MDP using the full p (including terminal zero)
        PS0, PS1, RS0, RS1, _ = build_mdp(p_full, C, K)
        
        # Create restless bandit model
        model1 = bandit.restless_bandit_from_P0P1_R0R1(PS0, PS1, RS0, RS1)
        
        # Discounted indices from the package
        pkg_disc_1 = model1.whittle_indices(discount=beta)
        print("\nSimple MDP discounted indices (beta = {}):".format(beta))
        for k in range(N):
            print(f"  State ({k}): W = {pkg_disc_1[k]:.6f}")

        
        # (Optional) Average‑cost indices (discount=0)
        pkg_avg = model.whittle_indices(discount=0.0)
        pkg_avg_passive = pkg_avg[1:N+1]
        pkg_avg_active_special = pkg_avg[0]
        
        print("\nPackage average‑cost indices:")
        print(f"  Active state (0,1): W = {pkg_avg_active_special:.4f}")
        for k, val in enumerate(pkg_avg_passive):
            print(f"  Passive state ({k},0): W = {val:.4f}")
        print("=" * 60)






# # Example usage
# N = 4
# seed = 42  # or any integer
# np.random.seed(seed)
# p = np.sort(np.random.uniform(0, 1.0, N))[::-1]
# p_full = np.append(p, 0.0)
# C, K = 5.0, 500.0
# C_switch = 1.0
# beta = 0.95

# print("="*70)
# print(f"Survival probabilities p[0..{N-1}]: {np.round(p,3)}")
# print("="*70)

# P0_ext, P1_ext, R0_ext, R1_ext, state_labels_ext = build_extended_mdp(N, p_full, C, K, C_switch)

# # Print matrices with state labels both vertically and horizontally
# print_transition_matrix(P0_ext, state_labels_ext, "Passive (a=0)")
# print_reward_vector(R0_ext, state_labels_ext, "Passive (a=0)")

# print_transition_matrix(P1_ext, state_labels_ext, "Active (a=1)")
# print_reward_vector(R1_ext, state_labels_ext, "Active (a=1)")