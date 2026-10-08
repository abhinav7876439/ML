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
    p : array of length N+1, survival probabilities p[0]..p[N] (p[N]=0 terminal)
    beta : discount factor (0 < beta < 1)
    
    Returns:
    G : array of length N+2, G[0]=0, G[k+1] = G[k] + (1-p[k]) * H[k]
    H : array of length N+1, H[k] = beta^(k+1) * prod_{i=0}^{k-1} p[i]
    """
    N = len(p)  # Now p has length N+1 (including terminal zero)
    H = np.zeros(N)
    G = np.zeros(N+1)          # G[0] = 0, up to G[N]
    prod = 1.0
    for k in range(N):
        # H[k] = beta^(k+1) * prod_{i=0}^{k-1} p[i]
        H[k] = beta**(k+1) * prod
        G[k+1] = G[k] + (1 - p[k]) * H[k]
        prod *= p[k]           # update product for next iteration
    return G, H

def closed_form_indices(p, beta, K, C, C_switch):
    """
    Compute closed-form Whittle indices for passive states (k,0) for k=0..N
    and the special active state (0,1).
    
    Returns:
    W_passive : array of length N+1, indices for k = 0..N
    W_active_special : float, index for state (0,1)
    """
    N = len(p)  # p has length N+1 including terminal zero
    G, H = compute_G_H(p, beta)
    
    # Passive states (k,0) for k = 0 to N
    W_passive = np.zeros(N)  # N = N+1 states? Let's clarify
    # Actually, we need N+1 passive states (from k=0 to N)
    # But since p[N]=0, we treat k=N separately
    W_passive = np.zeros(N)  # This will have length N+1
    for k in range(N):
        # Using formulas from the paper
        num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
        den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
        # Avoid division by zero (should not happen for valid parameters)
        if den != 0:
            W_passive[k] = K * (num / den) - C - C_switch
        else:
            W_passive[k] = np.inf
    
    # Special active state (0,1)
    if N >= 2:  # N here is the length of p (which is N+1 in original notation)
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
    p_full = np.append(p, 0.0)  # terminal zero for the extended MDP (now length N+1)
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
    
    # Closed-form indices
    W_passive, W_active = closed_form_indices(p_full, beta, K, C, C_switch)
    
    output_file.write("\nClosed-form Whittle indices (discounted, beta = {}):\n".format(beta))
    output_file.write(f"  Active state (0,1): W = {W_active:.6f}\n")
    # Print passive states from k=0 to N (total N+1 states)
    for k in range(N+1):  # N+1 passive states (0,0) through (N,0)
        output_file.write(f"  Passive state ({k},0): W = {W_passive[k]:.6f}\n")
    
    # Compare with markovianbandit package if available
    if PKG_AVAILABLE:
        # Create restless bandit model (states: 0=active, 1=N+1 passive)
        model = bandit.restless_bandit_from_P0P1_R0R1(P0_ext, P1_ext, R0_ext, R1_ext)
        
        # Discounted indices from the package
        pkg_disc = model.whittle_indices(discount=beta)
        # Package indices order: [active_state, passive_state_0, passive_state_1, ..., passive_state_N]
        pkg_active_special = pkg_disc[0]
        pkg_passive = pkg_disc[1:]  # This should have N+1 elements
        
        output_file.write("\nPackage discounted indices (beta = {}):\n".format(beta))
        output_file.write(f"  Active state (0,1): W = {pkg_active_special:.6f}\n")
        for k in range(N+1):
            output_file.write(f"  Passive state ({k},0): W = {pkg_passive[k]:.6f}\n")
        
        # Differences
        output_file.write("\nDifferences (closed-form - package):\n")
        output_file.write(f"  Active (0,1): {W_active - pkg_active_special:.2e}\n")
        for k in range(N+1):
            diff = W_passive[k] - pkg_passive[k]
            output_file.write(f"  Passive ({k},0): {diff:.2e}\n")
        
        # Average-cost indices (using discount=0)
        pkg_avg = model.whittle_indices(discount=0.0)
        pkg_avg_active_special = pkg_avg[0]
        pkg_avg_passive = pkg_avg[1:]
        
        output_file.write("\nPackage average-cost indices:\n")
        output_file.write(f"  Active state (0,1): W = {pkg_avg_active_special:.6f}\n")
        for k, val in enumerate(pkg_avg_passive):
            output_file.write(f"  Passive state ({k},0): W = {val:.6f}\n")
    
    output_file.write("\n" + "="*80 + "\n")
    output_file.flush()

# ============================================================================
# Main execution: Loop over different parameter values including beta
# ============================================================================
if __name__ == "__main__":
    # Fixed parameters
    N = 4
    
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
                                'p_values': str(np.round(p_full[:N+1], 4)),  # Save all N+1 probabilities
                                'W_active': W_active,
                            }
                            # Add passive states (k=0 to N)
                            for k in range(N+1):
                                summary_row[f'W_passive_{k}'] = W_passive[k] if k < len(W_passive) else np.nan
                            
                            # Add package results if available
                            if PKG_AVAILABLE:
                                P0_ext, P1_ext, R0_ext, R1_ext, _ = build_extended_mdp(N, p_full, C, K, C_switch)
                                model = bandit.restless_bandit_from_P0P1_R0R1(P0_ext, P1_ext, R0_ext, R1_ext)
                                pkg_disc = model.whittle_indices(discount=beta)
                                pkg_passive = pkg_disc[1:]  # All passive states (N+1 of them)
                                pkg_active_special = pkg_disc[0]
                                
                                summary_row['pkg_W_active'] = pkg_active_special
                                for k in range(N+1):
                                    summary_row[f'pkg_W_passive_{k}'] = pkg_passive[k] if k < len(pkg_passive) else np.nan
                                    summary_row[f'diff_passive_{k}'] = W_passive[k] - pkg_passive[k] if k < len(W_passive) and k < len(pkg_passive) else np.nan
                                summary_row['diff_active'] = W_active - pkg_active_special
                            
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
        for k in range(N+1):
            diff_col = f'diff_passive_{k}'
            if diff_col in summary_df.columns:
                print(f"  Passive state {k} - Mean absolute difference: {np.mean(np.abs(summary_df[diff_col])):.2e}")
    
    # Optional: Create some visualizations
    try:
        import matplotlib.pyplot as plt
        
        # Plot 1: Effect of beta on indices (including active state and all passive states)
        fixed_C = C_values[0]
        fixed_K = K_values[0]
        fixed_C_switch = C_switch_values[0]
        
        # Create a figure with rows for active and all passive states
        n_plots = N + 2  # Active + (N+1) passive states
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        # Filter data for fixed parameters
        mask = (summary_df['C'] == fixed_C) & (summary_df['K'] == fixed_K) & (summary_df['C_switch'] == fixed_C_switch)
        filtered_df = summary_df[mask]
        
        if len(filtered_df) > 0:
            # Plot 0: Active state
            ax = axes[0]
            for tm_idx in filtered_df['tm_idx'].unique():
                tm_data = filtered_df[filtered_df['tm_idx'] == tm_idx]
                ax.plot(tm_data['beta'], tm_data['W_active'], 'o-', label=f'TM {tm_idx}')
            ax.set_xlabel('Beta (discount factor)')
            ax.set_ylabel('Whittle Index')
            ax.set_title('Active State (0,1)')
            ax.legend()
            ax.grid(True)
            
            # Plot passive states
            for k in range(N+1):
                ax = axes[k+1]
                for tm_idx in filtered_df['tm_idx'].unique():
                    tm_data = filtered_df[filtered_df['tm_idx'] == tm_idx]
                    col_name = f'W_passive_{k}'
                    if col_name in tm_data.columns:
                        ax.plot(tm_data['beta'], tm_data[col_name], 'o-', label=f'TM {tm_idx}')
                ax.set_xlabel('Beta (discount factor)')
                ax.set_ylabel(f'Whittle Index')
                ax.set_title(f'Passive State ({k},0)')
                ax.legend()
                ax.grid(True)
            
            # Hide extra subplots if any
            for idx in range(len(axes)):
                if idx > n_plots - 1:
                    axes[idx].set_visible(False)
            
            plt.suptitle(f'Effect of Beta on Whittle Indices (C={fixed_C}, K={fixed_K}, C_switch={fixed_C_switch})')
            plt.tight_layout()
            plt.savefig(f'whittle_vs_beta_{timestamp}.png', dpi=150)
            print(f"\nPlot saved: whittle_vs_beta_{timestamp}.png")
        
        # Plot 2: Compare closed-form vs package for active state across different betas
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