import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def build_extended_mdp(N, p, C, K, C_switch):
    """Build the extended MDP for the restless bandit problem"""
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

def value_iteration(P0, P1, R0, R1, w_val, discount, max_iter=10000, tol=1e-8):
    """
    Value iteration for discounted MDP with subsidy w for active action.
    Returns the optimal value function V(s) for all states.
    """
    N_states = P0.shape[0]
    V = np.zeros(N_states)
    
    for iteration in range(max_iter):
        V_old = V.copy()
        
        for s in range(N_states):
            # Q-values for each action
            Q0 = R0[s] + discount * np.dot(P0[s, :], V_old)  # Passive action
            Q1 = (R1[s] - w_val) + discount * np.dot(P1[s, :], V_old)  # Active action with subsidy
            V[s] = min(Q0, Q1)  # Minimizing cost
        
        # Check convergence
        if np.max(np.abs(V - V_old)) < tol:
            break
    
    return V

def compute_whittle_index_binary_search(state_idx, P0, P1, R0, R1, discount, 
                                       w_min=None, w_max=None, n_iter=50, verbose=False):
    """
    Compute Whittle index for a single state using binary search.
    The Whittle index is the w that makes Q0 = Q1.
    """
    # Auto-detect reasonable bounds if not provided
    if w_min is None or w_max is None:
        # Test extreme values to find bounds
        w_test = 0
        V = value_iteration(P0, P1, R0, R1, w_test, discount)
        Q0 = R0[state_idx] + discount * np.dot(P0[state_idx, :], V)
        Q1 = (R1[state_idx] - w_test) + discount * np.dot(P1[state_idx, :], V)
        
        if Q0 < Q1:  # Passive preferred at w=0, need lower w to make active attractive
            w_min, w_max = -100, 0
        else:  # Active preferred at w=0, need higher w to make passive attractive
            w_min, w_max = 0, 100
        
        # Expand bounds if needed
        for _ in range(5):
            V = value_iteration(P0, P1, R0, R1, w_min, discount)
            Q0 = R0[state_idx] + discount * np.dot(P0[state_idx, :], V)
            Q1 = (R1[state_idx] - w_min) + discount * np.dot(P1[state_idx, :], V)
            if Q0 < Q1:  # Still passive preferred at lower bound
                w_min -= 50
            else:
                break
                
        for _ in range(5):
            V = value_iteration(P0, P1, R0, R1, w_max, discount)
            Q0 = R0[state_idx] + discount * np.dot(P0[state_idx, :], V)
            Q1 = (R1[state_idx] - w_max) + discount * np.dot(P1[state_idx, :], V)
            if Q0 > Q1:  # Still active preferred at upper bound
                w_max += 50
            else:
                break
    
    # Binary search for indifference point
    for iteration in range(n_iter):
        w_mid = (w_min + w_max) / 2
        
        # Compute value function with current w
        V = value_iteration(P0, P1, R0, R1, w_mid, discount)
        
        # Compute Q-values for the state
        Q0 = R0[state_idx] + discount * np.dot(P0[state_idx, :], V)
        Q1 = (R1[state_idx] - w_mid) + discount * np.dot(P1[state_idx, :], V)
        
        if verbose and iteration % 10 == 0:
            print(f"    Iter {iteration}: w={w_mid:.4f}, Q0={Q0:.4f}, Q1={Q1:.4f}, diff={Q0-Q1:.4f}")
        
        # Adjust bounds based on which action is better
        if Q0 < Q1:  # Passive is better (lower cost)
            w_max = w_mid  # Need to decrease w to make active more attractive
        else:  # Active is better (or equal)
            w_min = w_mid  # Need to increase w to make passive more attractive
    
    return (w_min + w_max) / 2

def compute_all_whittle_indices(P0, P1, R0, R1, discount, w_range=None, verbose=True):
    """
    Compute Whittle indices for all states using binary search.
    Returns array of indices for each state.
    """
    N_states = P0.shape[0]
    indices = np.zeros(N_states)
    
    print(f"\n{'='*70}")
    print(f"COMPUTING WHITTLE INDICES FOR {N_states} STATES")
    print(f"{'='*70}")
    
    for s in range(N_states):
        if verbose:
            print(f"\nState {s}: Computing index...")
        
        indices[s] = compute_whittle_index_binary_search(
            s, P0, P1, R0, R1, discount, 
            w_min=w_range[0] if w_range else None, 
            w_max=w_range[1] if w_range else None,
            verbose=verbose
        )
        
        if verbose:
            print(f"  ✓ Index = {indices[s]:.6f}")
    
    return indices

def verify_indices(P0, P1, R0, R1, indices, discount, tol=1e-6):
    """
    Verify that computed indices indeed make Q0 ≈ Q1.
    """
    print(f"\n{'='*70}")
    print(f"VERIFYING WHITTLE INDICES")
    print(f"{'='*70}")
    
    for s, w in enumerate(indices):
        V = value_iteration(P0, P1, R0, R1, w, discount)
        Q0 = R0[s] + discount * np.dot(P0[s, :], V)
        Q1 = (R1[s] - w) + discount * np.dot(P1[s, :], V)
        diff = abs(Q0 - Q1)
        
        status = "✓" if diff < tol else "⚠"
        print(f"State {s:2d}: w={w:10.6f}, Q0={Q0:10.6f}, Q1={Q1:10.6f}, diff={diff:.2e} {status}")
    
    return True

def analyze_policy(P0, P1, R0, R1, indices, discount, w_multipliers=[0.5, 1.0, 1.5]):
    """
    Analyze optimal policy for different w values relative to indices.
    """
    print(f"\n{'='*70}")
    print(f"POLICY ANALYSIS AT DIFFERENT SUBSIDY LEVELS")
    print(f"{'='*70}")
    
    N_states = P0.shape[0]
    
    for mult in w_multipliers:
        print(f"\n--- w = {mult} × Index ---")
        for s in range(N_states):
            w = indices[s] * mult
            V = value_iteration(P0, P1, R0, R1, w, discount)
            Q0 = R0[s] + discount * np.dot(P0[s, :], V)
            Q1 = (R1[s] - w) + discount * np.dot(P1[s, :], V)
            
            optimal_action = "Passive" if Q0 < Q1 else "Active"
            print(f"State {s:2d}: w={w:8.3f}, Optimal={optimal_action:7s} (Q0={Q0:8.3f}, Q1={Q1:8.3f})")

def visualize_results(indices, state_labels, P0, P1, R0, R1, discount):
    """
    Create comprehensive visualization of Whittle indices.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Whittle indices for all states
    ax = axes[0, 0]
    x_pos = np.arange(len(indices))
    colors = ['red' if '(0,1)' in label else 'blue' for label in state_labels]
    bars = ax.bar(x_pos, indices, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('State Index')
    ax.set_ylabel('Whittle Index')
    ax.set_title('Whittle Indices for All States')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(state_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, idx in zip(bars, indices):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{idx:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Difference between consecutive states
    ax = axes[0, 1]
    diffs = np.diff(indices)
    ax.plot(range(1, len(indices)), diffs, 'o-', color='green', markersize=6, linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('State Transition')
    ax.set_ylabel('Index Difference')
    ax.set_title('Consecutive State Index Differences')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Q-value equality verification
    ax = axes[1, 0]
    diffs_verification = []
    for s, w in enumerate(indices):
        V = value_iteration(P0, P1, R0, R1, w, discount)
        Q0 = R0[s] + discount * np.dot(P0[s, :], V)
        Q1 = (R1[s] - w) + discount * np.dot(P1[s, :], V)
        diffs_verification.append(Q0 - Q1)
    
    ax.bar(range(len(indices)), diffs_verification, alpha=0.7, color='purple')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('State')
    ax.set_ylabel('Q0 - Q1')
    ax.set_title(f'Verification: Q0 - Q1 at Index (should be near 0)')
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(state_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Index sensitivity analysis
    ax = axes[1, 1]
    # Test sensitivity to discount factor
    discount_factors = np.linspace(0.8, 0.99, 10)
    indices_sensitivity = []
    
    for beta in discount_factors:
        indices_temp = []
        for s in range(min(5, len(indices))):  # First 5 states only
            idx = compute_whittle_index_binary_search(s, P0, P1, R0, R1, beta, 
                                                      n_iter=30, verbose=False)
            indices_temp.append(idx)
        indices_sensitivity.append(indices_temp)
    
    indices_sensitivity = np.array(indices_sensitivity)
    for i in range(indices_sensitivity.shape[1]):
        ax.plot(discount_factors, indices_sensitivity[:, i], 'o-', 
                label=state_labels[i], linewidth=2, markersize=4)
    
    ax.set_xlabel('Discount Factor β')
    ax.set_ylabel('Whittle Index')
    ax.set_title('Sensitivity to Discount Factor')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('whittle_indices_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Figure saved as 'whittle_indices_analysis.png'")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Parameters
    N = 4
    np.random.seed(42)
    p = np.sort(np.random.uniform(0.3, 0.95, N))[::-1]  # Decreasing survival probabilities
    p_full = np.append(p, 0.0)  # Terminal zero
    C, K = 5.0, 100.0
    C_switch = 2.0
    discount = 0.95
    
    print("="*70)
    print("RESTLESS BANDIT WHITTLE INDICES - VALUE ITERATION + BINARY SEARCH")
    print("="*70)
    
    print(f"\nParameters:")
    print(f"  N = {N} (states: 0 to {N})")
    print(f"  Survival probabilities: {np.round(p, 4)}")
    print(f"  Repair cost C = {C}")
    print(f"  Failure cost K = {K}")
    print(f"  Switching cost C_switch = {C_switch}")
    print(f"  Discount factor β = {discount}")
    
    # Build extended MDP
    P0, P1, R0, R1, state_labels = build_extended_mdp(N, p_full, C, K, C_switch)
    print(f"\nNumber of states: {P0.shape[0]}")
    print(f"State labels: {state_labels}")
    
    # Compute Whittle indices
    indices = compute_all_whittle_indices(P0, P1, R0, R1, discount, w_range=(-50, 50))
    
    # Display results
    print(f"\n{'='*70}")
    print(f"WHITTLE INDICES RESULTS")
    print(f"{'='*70}")
    print(f"\n{'State':<15} {'Index Value':<15} {'Rank':<10}")
    print("-"*40)
    
    for i, (label, idx) in enumerate(zip(state_labels, indices)):
        rank = np.argsort(np.argsort(indices))[i] + 1
        print(f"{label:<15} {idx:>12.6f}   {rank:>2}")
    
    # Verify indices
    verify_indices(P0, P1, R0, R1, indices, discount, tol=1e-4)
    
    # Analyze policy at different subsidy levels
    analyze_policy(P0, P1, R0, R1, indices, discount, w_multipliers=[0.8, 1.0, 1.2])
    
    # Visualize results
    visualize_results(indices, state_labels, P0, P1, R0, R1, discount)
    
    # Additional analysis: Compare active vs passive states
    print(f"\n{'='*70}")
    print(f"ACTIVE VS PASSIVE STATE COMPARISON")
    print(f"{'='*70}")
    
    for i, label in enumerate(state_labels):
        if '(0,1)' in label:
            print(f"\nSpecial Active State {label}:")
            print(f"  Index = {indices[i]:.6f}")
            print(f"  Note: This state has a different cost structure (no switching cost)")
        elif ',0)' in label:
            x = int(label.split(',')[0].strip('('))
            print(f"\nPassive State {label}:")
            print(f"  Index = {indices[i]:.6f}")
            if x > 0:
                active_comparison = f"  Compare with active state ({x},1) - would have index = {indices[i] + C_switch:.6f} (theoretical)"
                print(active_comparison)
    
    print("\n" + "="*70)
    print("✓ COMPUTATION COMPLETED SUCCESSFULLY")
    print("="*70)