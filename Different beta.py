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



        # # Build the simple MDP using the full p (including terminal zero)
        # PS0, PS1, RS0, RS1, _ = build_mdp(p, C, K)
        
        # # Create restless bandit model
        # model1 = bandit.restless_bandit_from_P0P1_R0R1(PS0, PS1, RS0, RS1)

        # # Discounted indices from the package
        # pkg_disc_1 = model1.whittle_indices(discount=beta)


        # output_file.write("\n (Simple MDP) Package discounted indices (beta = {}):\n".format(beta))
        # for k in range(N):
        #     output_file.write(f"  State ({k}): W = {pkg_disc_1[k]:.6f}\n")


        
        # # (Optional) Average‑cost indices (discount=0)
        # pkg_avg_1 = model1.whittle_indices(discount=0.0)
        
        # output_file.write("\n (Simple MDP) Package average‑cost indices:\n")
        # for k in range(N):
        #     output_file.write(f"  State ({k}): W = {pkg_avg_1[k]:.6f}\n")
        # output_file.write("=" * 60)
    
    output_file.write("\n" + "="*80 + "\n")
    output_file.flush()

# ============================================================================
# Main execution: Loop over different parameter values including beta
# ============================================================================
if __name__ == "__main__":
    # Fixed parameters
    N = 4
    np.random.seed(42)
    # Parameter grids to explore
    C_values = [5.0, 10.0, 20.0]              # Different activation costs
    K_values = [500.0, 1000.0, 2000.0]         # Different penalty costs
    C_switch_values = [0.0, 5.0, 10.0, 20.0, 40.0, 200.0, 500.0]         # Different switching costs (including 0)
    beta_values = [0.9, 0.95, 0.99]            # Different discount factors
    
    # Number of different transition matrices to try
    num_transition_matrices = 3
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"whittle_indices_comparison_{timestamp}.txt"
    
    # Also create a summary CSV file for easier analysis
    summary_data = []
    
    # Open output file
    with open(output_filename, 'w') as output_file:
        output_file.write("="*80 + "\n")
        output_file.write("WHITTLE INDICES COMPARISON: CLOSED-FORM vs PACKAGE\n")
        output_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write("="*80 + "\n\n")
        
        output_file.write("Parameter ranges explored:\n")
        output_file.write(f"  C (activation cost): {C_values}\n")
        output_file.write(f"  K (penalty cost): {K_values}\n")
        output_file.write(f"  C_switch (switching cost): {C_switch_values}\n")
        output_file.write(f"  beta (discount factor): {beta_values}\n")
        output_file.write(f"  Number of transition matrices: {num_transition_matrices}\n")
        output_file.write(f"  Total combinations: {len(C_values) * len(K_values) * len(C_switch_values) * len(beta_values) * num_transition_matrices}\n")
        output_file.write("\n" + "="*80 + "\n")
        
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
                        for beta in beta_values:
                            instance_counter += 1
                            
                            # Store summary info
                            W_passive, W_active = closed_form_indices(p_full, beta, K, C, C_switch)
                            
                            summary_row = {
                                'instance': instance_counter,
                                'tm_idx': tm_idx + 1,
                                'C': C,
                                'K': K,
                                'C_switch': C_switch,
                                'beta': beta,
                                'p_values': str(np.round(p_full[:N], 4)),
                                'W_active': W_active,
                                'W_passive_0': W_passive[0] if len(W_passive) > 0 else np.nan,
                                'W_passive_1': W_passive[1] if len(W_passive) > 1 else np.nan,
                                'W_passive_2': W_passive[2] if len(W_passive) > 2 else np.nan,
                                'W_passive_3': W_passive[3] if len(W_passive) > 3 else np.nan,
                                'W_passive_4': W_passive[4] if len(W_passive) > 4 else np.nan,
                            }
                            
                            # Add package results if available
                            if PKG_AVAILABLE:
                                P0, P1, R0, R1, _ = build_extended_mdp(N, p_full, C, K, C_switch)
                                model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
                                pkg_disc = model.whittle_indices(discount=beta)
                                pkg_passive = pkg_disc[1:N+2]
                                pkg_active_special = pkg_disc[0]
                                
                                summary_row['pkg_W_active'] = pkg_active_special
                                summary_row['pkg_W_passive_0'] = pkg_passive[0] if len(pkg_passive) > 0 else np.nan
                                summary_row['pkg_W_passive_1'] = pkg_passive[1] if len(pkg_passive) > 1 else np.nan
                                summary_row['pkg_W_passive_2'] = pkg_passive[2] if len(pkg_passive) > 2 else np.nan
                                summary_row['pkg_W_passive_3'] = pkg_passive[3] if len(pkg_passive) > 3 else np.nan
                                summary_row['pkg_W_passive_4'] = pkg_passive[4] if len(pkg_passive) > 4 else np.nan
                                summary_row['diff_active'] = W_active - pkg_active_special
                                summary_row['diff_passive_0'] = W_passive[0] - pkg_passive[0] if len(W_passive) > 0 else np.nan
                            
                            summary_data.append(summary_row)
                            
                            # Run full comparison
                            run_comparison(N, p, p_full, C, K, C_switch, beta, 
                                         instance_counter, output_file)
        
        output_file.write("\n" + "="*80 + "\n")
        output_file.write(f"COMPARISON COMPLETE\n")
        output_file.write(f"Total instances processed: {instance_counter}\n")
        output_file.write("="*80 + "\n")
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv = f"whittle_indices_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Detailed results: {output_filename}")
    print(f"  Summary CSV: {summary_csv}")
    print(f"\nTotal instances processed: {instance_counter}")
    
    # Print summary to console
    print("\nParameter ranges explored:")
    print(f"  C (activation cost): {C_values}")
    print(f"  K (penalty cost): {K_values}")
    print(f"  C_switch (switching cost): {C_switch_values}")
    print(f"  beta (discount factor): {beta_values}")
    print(f"  Number of transition matrices: {num_transition_matrices}")
    print(f"  Total combinations: {len(C_values) * len(K_values) * len(C_switch_values) * len(beta_values) * num_transition_matrices}")
    
    # Print some statistics
    if PKG_AVAILABLE and len(summary_data) > 0:
        print("\nAccuracy Statistics (closed-form vs package):")
        print(f"  Active state - Mean absolute difference: {np.mean(np.abs(summary_df['diff_active'])):.2e}")
        print(f"  Passive state 0 - Mean absolute difference: {np.mean(np.abs(summary_df['diff_passive_0'])):.2e}")
        if 'diff_passive_1' in summary_df.columns:
            print(f"  Passive state 1 - Mean absolute difference: {np.mean(np.abs(summary_df['diff_passive_1'])):.2e}")
    
# Optional: Create some visualizations
try:
    import matplotlib.pyplot as plt
    
    # Plot 1: Effect of beta on indices (including active state)
    fixed_C = C_values[0]
    fixed_K = K_values[0]
    fixed_C_switch = C_switch_values[0]
    
    # Create a figure with 2 rows, 3 columns (to include active state + 4 passive states)
    # Or better: separate figure for active state
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Filter data for fixed parameters
    mask = (summary_df['C'] == fixed_C) & (summary_df['K'] == fixed_K) & (summary_df['C_switch'] == fixed_C_switch)
    filtered_df = summary_df[mask]
    
    if len(filtered_df) > 0:
        # Plot 1: Active state
        ax = axes[0]
        for tm_idx in filtered_df['tm_idx'].unique():
            tm_data = filtered_df[filtered_df['tm_idx'] == tm_idx]
            ax.plot(tm_data['beta'], tm_data['W_active'], 'o-', label=f'TM {tm_idx}')
        ax.set_xlabel('Beta (discount factor)')
        ax.set_ylabel('Whittle Index')
        ax.set_title('Active State (0,1)')
        ax.legend()
        ax.grid(True)
        
        # Plot 2-5: Passive states
        for i, col in enumerate(['W_passive_0', 'W_passive_1', 'W_passive_2', 'W_passive_3', 'W_passive_4']):
            if col in filtered_df.columns:
                ax = axes[i+1]
                for tm_idx in filtered_df['tm_idx'].unique():
                    tm_data = filtered_df[filtered_df['tm_idx'] == tm_idx]
                    ax.plot(tm_data['beta'], tm_data[col], 'o-', label=f'TM {tm_idx}')
                ax.set_xlabel('Beta (discount factor)')
                ax.set_ylabel(f'Whittle Index')
                ax.set_title(f'Passive State ({i},0)')
                ax.legend()
                ax.grid(True)
        
        # Hide the 6th subplot if not needed
        axes[5].set_visible(False)
        
        plt.suptitle(f'Effect of Beta on Whittle Indices (C={fixed_C}, K={fixed_K}, C_switch={fixed_C_switch})')
        plt.tight_layout()
        plt.savefig(f'whittle_vs_beta_{timestamp}.png', dpi=150)
        print(f"\nPlot saved: whittle_vs_beta_{timestamp}.png")
    
    # Plot 2: Effect of switching cost on indices (including active state)
    fixed_beta = beta_values[1]  # middle beta value
    fixed_K = K_values[0]
    fixed_C = C_values[0]
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes2 = axes2.flatten()
    
    mask2 = (summary_df['beta'] == fixed_beta) & (summary_df['K'] == fixed_K) & (summary_df['C'] == fixed_C)
    filtered_df2 = summary_df[mask2]
    
    if len(filtered_df2) > 0:
        # Active state plot
        ax = axes2[0]
        for tm_idx in filtered_df2['tm_idx'].unique():
            tm_data = filtered_df2[filtered_df2['tm_idx'] == tm_idx]
            ax.plot(tm_data['C_switch'], tm_data['W_active'], 'o-', label=f'TM {tm_idx}')
        ax.set_xlabel('Switching Cost (C_switch)')
        ax.set_ylabel('Whittle Index')
        ax.set_title('Active State (0,1)')
        ax.legend()
        ax.grid(True)
        
        # Passive states plots
        for i, col in enumerate(['W_passive_0', 'W_passive_1', 'W_passive_2', 'W_passive_3', 'W_passive_4']):
            if col in filtered_df2.columns:
                ax = axes2[i+1]
                for tm_idx in filtered_df2['tm_idx'].unique():
                    tm_data = filtered_df2[filtered_df2['tm_idx'] == tm_idx]
                    ax.plot(tm_data['C_switch'], tm_data[col], 'o-', label=f'TM {tm_idx}')
                ax.set_xlabel('Switching Cost (C_switch)')
                ax.set_ylabel(f'Whittle Index')
                ax.set_title(f'Passive State ({i},0)')
                ax.legend()
                ax.grid(True)
        
        axes2[5].set_visible(False)
        
        plt.suptitle(f'Effect of Switching Cost on Whittle Indices (beta={fixed_beta}, C={fixed_C}, K={fixed_K})')
        plt.tight_layout()
        plt.savefig(f'whittle_vs_switchcost_{timestamp}.png', dpi=150)
        print(f"Plot saved: whittle_vs_switchcost_{timestamp}.png")
        
    # Plot 3: Compare closed-form vs package for active state across different betas
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    for tm_idx in filtered_df['tm_idx'].unique():
        tm_data = filtered_df[filtered_df['tm_idx'] == tm_idx]
        ax3.plot(tm_data['beta'], tm_data['diff_active'], 'o-', label=f'TM {tm_idx} - Active State')
    
    ax3.set_xlabel('Beta (discount factor)')
    ax3.set_ylabel('Difference (Closed-form - Package)')
    ax3.set_title('Active State: Closed-form vs Package Difference')
    ax3.legend()
    ax3.grid(True)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'active_state_diff_vs_beta_{timestamp}.png', dpi=150)
    print(f"Plot saved: active_state_diff_vs_beta_{timestamp}.png")
    
except Exception as e:
    print(f"\nNote: Could not create plots: {e}")