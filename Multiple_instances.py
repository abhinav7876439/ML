import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import bisect
import networkx as nx
from scipy.optimize import brentq
from datetime import datetime
import os

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
    if N >= 2:
        G1 = G[1]
        H1 = H[1]
    else:
        G1 = G[1]
        H1 = 0.0
    
    numerator = (1 - beta) * (K * G1 + C_switch * H1)
    denominator = beta * (1 - G1) - H1
    W_active = (numerator / denominator) - C if denominator != 0 else np.inf
    
    return W_passive, W_active

def generate_transition_matrix(N, random_state=None):
    """Generate random survival probabilities."""
    if random_state is not None:
        np.random.seed(random_state)
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]  # decreasing survival probabilities
    p_full = np.append(p, 0.0)  # terminal zero for the extended MDP
    return p, p_full

def run_comparison(N, p, p_full, C, K, C_switch, beta, instance_num, output_file):
    """Run comparison for a single parameter set and write results to file."""
    
    output_file.write("\n" + "="*80 + "\n")
    output_file.write(f"INSTANCE {instance_num}\n")
    output_file.write("="*80 + "\n")
    output_file.write(f"Parameters:\n")
    output_file.write(f"  N = {N}\n")
    output_file.write(f"  Survival probabilities p[0..{N}]: {np.round(p_full, 4)}\n")
    output_file.write(f"  C = {C}\n")
    output_file.write(f"  K = {K}\n")
    output_file.write(f"  C_switch = {C_switch}\n")
    output_file.write(f"  beta = {beta}\n")
    output_file.write("-"*80 + "\n")
    
    # Build MDP
    P0_ext, P1_ext, R0_ext, R1_ext, state_labels_ext = build_extended_mdp(N, p_full, C, K, C_switch)
    
    # Print transition matrices (optional - can be commented out for cleaner output)
    output_file.write("\nTransition matrix for action Passive (a=0):\n")
    df0 = pd.DataFrame(P0_ext, index=state_labels_ext, columns=state_labels_ext)
    output_file.write(df0.to_string() + "\n")
    
    output_file.write("\nReward vector for action Passive (a=0):\n")
    output_file.write(str(R0_ext) + "\n")
    
    output_file.write("\nTransition matrix for action Active (a=1):\n")
    df1 = pd.DataFrame(P1_ext, index=state_labels_ext, columns=state_labels_ext)
    output_file.write(df1.to_string() + "\n")
    
    output_file.write("\nReward vector for action Active (a=1):\n")
    output_file.write(str(R1_ext) + "\n")
    
    # Closed-form indices
    W_passive, W_active = closed_form_indices(p, beta, K, C, C_switch)
    
    output_file.write("\nClosed-form Whittle indices (discounted, beta = {}):\n".format(beta))
    output_file.write(f"  Active state (0,1): W = {W_active:.6f}\n")
    for k in range(N):
        output_file.write(f"  Passive state ({k},0): W = {W_passive[k]:.6f}\n")
    
    # Compare with markovianbandit package if available
    if PKG_AVAILABLE:
        P0, P1, R0, R1, _ = build_extended_mdp(N, p_full, C, K, C_switch)
        
        # Create restless bandit model
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
        
        # Discounted indices from the package
        pkg_disc = model.whittle_indices(discount=beta)
        pkg_passive = pkg_disc[1:N+1]
        pkg_active_special = pkg_disc[0]
        
        output_file.write("\nPackage discounted indices (beta = {}):\n".format(beta))
        output_file.write(f"  Active state (0,1): W = {pkg_active_special:.6f}\n")
        for k in range(N):
            output_file.write(f"  Passive state ({k},0): W = {pkg_passive[k]:.6f}\n")
        
        # Differences
        output_file.write("\nDifferences (closed-form - package):\n")
        output_file.write(f"  Active (0,1): {W_active - pkg_active_special:.2e}\n")
        for k in range(N):
            diff = W_passive[k] - pkg_passive[k]
            output_file.write(f"  Passive ({k},0): {diff:.2e}\n")
        
        # Average-cost indices
        pkg_avg = model.whittle_indices(discount=0.0)
        pkg_avg_passive = pkg_avg[1:N+1]
        pkg_avg_active_special = pkg_avg[0]
        
        output_file.write("\nPackage average-cost indices:\n")
        output_file.write(f"  Active state (0,1): W = {pkg_avg_active_special:.6f}\n")
        for k, val in enumerate(pkg_avg_passive):
            output_file.write(f"  Passive state ({k},0): W = {val:.6f}\n")
    
    output_file.write("\n" + "="*80 + "\n")
    output_file.flush()

# ============================================================================
# Main execution: Loop over different parameter values
# ============================================================================
if __name__ == "__main__":
    # Fixed parameters
    N = 4
    beta = 0.95
    
    # Parameter grids to explore
    C_values = [5.0, 10.0, 20.0]           # Different activation costs
    K_values = [500.0, 1000.0, 2000.0]      # Different penalty costs
    C_switch_values = [0.0, 5.0, 10.0]      # Different switching costs (including 0)
    
    # Number of different transition matrices to try
    num_transition_matrices = 3
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"whittle_indices_comparison_{timestamp}.txt"
    
    # Open output file
    with open(output_filename, 'w') as output_file:
        output_file.write("="*80 + "\n")
        output_file.write("WHITTLE INDICES COMPARISON: CLOSED-FORM vs PACKAGE\n")
        output_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write("="*80 + "\n\n")
        
        instance_counter = 0
        
        # Loop over different transition matrices (random seeds)
        for tm_idx in range(num_transition_matrices):
            # Generate different transition matrix for each iteration
            # No fixed seed - using different random values each time
            p, p_full = generate_transition_matrix(N, random_state=None)
            
            output_file.write("\n" + "#"*80 + "\n")
            output_file.write(f"TRANSITION MATRIX {tm_idx + 1}\n")
            output_file.write(f"Survival probabilities: {np.round(p_full, 4)}\n")
            output_file.write("#"*80 + "\n")
            
            # Loop over all parameter combinations
            for C in C_values:
                for K in K_values:
                    for C_switch in C_switch_values:
                        instance_counter += 1
                        run_comparison(N, p, p_full, C, K, C_switch, beta, 
                                     instance_counter, output_file)
        
        output_file.write("\n" + "="*80 + "\n")
        output_file.write(f"COMPARISON COMPLETE\n")
        output_file.write(f"Total instances processed: {instance_counter}\n")
        output_file.write("="*80 + "\n")
    
    print(f"\nResults saved to: {output_filename}")
    print(f"Total instances processed: {instance_counter}")
    
    # Print summary to console
    print("\nParameter ranges explored:")
    print(f"  C (activation cost): {C_values}")
    print(f"  K (penalty cost): {K_values}")
    print(f"  C_switch (switching cost): {C_switch_values}")
    print(f"  Number of transition matrices: {num_transition_matrices}")
    print(f"  Total combinations: {len(C_values) * len(K_values) * len(C_switch_values) * num_transition_matrices}")