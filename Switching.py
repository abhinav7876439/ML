import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import bisect, brentq
import networkx as nx
from scipy.stats import rankdata

# Try to import markovianbandit; if not available, skip package comparison
try:
    import markovianbandit as bandit
    PKG_AVAILABLE = True
except ImportError:
    PKG_AVAILABLE = False
    print("markovianbandit not installed; skip package comparison.")

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



def closed_form_indices_discounted(p, beta, K, C, C_switch):
    """Discounted Whittle indices for states x = 0..N-1 (active states)."""
    N = len(p)
    G, H = compute_G_H(p, beta)
    W = np.zeros(N)
    for k in range(N):
        num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
        den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
        if den != 0:
            W[k] = K * (num / den) - C - C_switch
        else:
            W[k] = np.inf
    return W




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

def closed_form_indices_discounted(p, beta, K, C, C_switch): 
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
    W_active = numerator / denominator if denominator != 0 else np.inf
    
    return W_passive, W_active

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



def get_optimal_policy(P0, P1, R0, R1, w_val, discount):
    """Get optimal policy for given intervention charge"""
    V = value_iteration(P0, P1, R0, R1, w_val, discount)
    
    policy = {}
    for s in range(P0.shape[0]):
        Q_passive = R0[s] + discount * np.dot(P0[s, :], V)
        Q_active = (R1[s] - w_val) + discount * np.dot(P1[s, :], V)
        policy[s] = 0 if Q_passive < Q_active else 1  # 0=passive, 1=active
    
    return policy

def state_to_label(s, state_labels):
    """Convert state index to label"""
    return state_labels[s] if s < len(state_labels) else f"State {s}"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Parameters
    N = 5
    np.random.seed(42)
    p = np.sort(np.random.uniform(0.3, 0.95, N))[::-1]  # decreasing survival probabilities
    p_full = np.append(p, 0.0)  # terminal zero for the extended MDP
    C, K = 10.0, 100.0
    C_switch = 5.0
    discount = 0.95
    
    print("=" * 80)
    print("RESTLESS BANDIT WHITTLE INDICES - COMPLETE ANALYSIS")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  N = {N} states")
    print(f"  Survival probabilities: {np.round(p_full, 4)}")
    print(f"  Repair cost C = {C}")
    print(f"  Failure cost K = {K}")
    print(f"  Switching cost C_switch = {C_switch}")
    print(f"  Discount factor β = {discount}")
    
    # Build MDP
    P0, P1, R0, R1, state_labels = build_extended_mdp(N, p_full, C, K, C_switch)
    N_states = P0.shape[0]


    # Print matrices with state labels both vertically and horizontally
    print_transition_matrix(P0, state_labels, "Passive (a=0)")
    print_reward_vector(R0, state_labels, "Passive (a=0)")

    print_transition_matrix(P1, state_labels, "Active (a=1)")
    print_reward_vector(R1, state_labels, "Active (a=1)")
    
    # 1. Compute Whittle indices using binary search
    print("\n" + "=" * 80)
    print("METHOD 1: BINARY SEARCH WITH VALUE ITERATION")
    print("=" * 80)
    
    whittle_indices_binary = compute_all_whittle_indices(P0, P1, R0, R1, discount)
    # def compute_all_whittle_indices(P0, P1, R0, R1, discount, w_range=None, verbose=True):
    print("\nComputed Whittle indices (binary search):")
    for s, w in enumerate(whittle_indices_binary):
        print(f"  State {state_to_label(s, state_labels):<10}: W = {w:.6f}")
    
    # 2. Compute closed-form indices
    print("\n" + "=" * 80)
    print("METHOD 2: CLOSED-FORM EXPRESSIONS")
    print("=" * 80)
    
    W_passive, W_active = closed_form_indices_discounted(p, discount, K, C, C_switch)
    
    print("\nClosed-form Whittle indices (discounted, β = {}):".format(discount))
    print(f"  Active state (0,1): W = {W_active:.6f}")
    for k in range(N): 
        print(f"  Passive state ({k},0): W = {W_passive[k]:.6f}")
    
    # 3. Compare with package if available
    if PKG_AVAILABLE:
        print("\n" + "=" * 80)
        print("METHOD 3: MARKOVIANBANDIT PACKAGE")
        print("=" * 80)
        
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
        pkg_indices = model.whittle_indices(discount=discount)
        
        # Package indices: [active_special, passive_0, passive_1, ...]
        pkg_active_special = pkg_indices[0]
        pkg_passive = pkg_indices[1:N+2]
    else:
        pkg_active_special = None
        pkg_passive = None

    print("\nPackage discounted indices (β = {}):".format(discount))
    if PKG_AVAILABLE:
        print(f"  Active state (0,1): W = {pkg_active_special:.6f}")
        for k in range(N):
            print(f"  Passive state ({k},0): W = {pkg_passive[k]:.6f}")
    else:
        print("  Package indices not available.")

    # 4. Create comprehensive comparison table
    print("\n" + "=" * 80)
    print("COMPARISON OF WHITTLE INDICES")
    print("=" * 80)
    
    # Create results dataframe
    results = []
    for i, label in enumerate(state_labels):
        if label == "(0,1)":
            binary_val = whittle_indices_binary[i]
            closed_val = W_active if label == "(0,1)" else np.nan
            pkg_val = pkg_active_special if PKG_AVAILABLE else None
        else:  # passive states
            idx = int(label.split(',')[0].strip('('))
            binary_val = whittle_indices_binary[i]
            closed_val = W_passive[idx] if idx < len(W_passive) else np.nan
            pkg_val = pkg_passive[idx] if (PKG_AVAILABLE and idx < len(pkg_passive)) else None
        
        results.append({
            'State': label,
            'Binary Search': binary_val,
            'Closed-Form': closed_val,
            'Package': pkg_val if PKG_AVAILABLE else 'N/A'
        })
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    if PKG_AVAILABLE:
        print("\nDifferences (Binary - Package):")
        for i, label in enumerate(state_labels):
            diff = whittle_indices_binary[i] - (pkg_active_special if i == 0 else pkg_passive[i-1])
            print(f"  {label}: {diff:.6f} ({diff:.2e})")
    
    # 5. Policy analysis at different intervention charges
    print("\n" + "=" * 80)
    print("POLICY ANALYSIS AT DIFFERENT INTERVENTION CHARGES")
    print("=" * 80)
    
    w_values = np.linspace(-20, 50, 7)
    print(f"\n{'w':<10} {'Optimal Actions (Active|Passive)':<50}")
    print("-" * 60)
    
    for w_val in w_values:
        policy = get_optimal_policy(P0, P1, R0, R1, w_val, discount)
        policy_str = []
        for i, label in enumerate(state_labels):
            action = "Repair" if policy[i] == 1 else "Degrade"
            policy_str.append(f"{label}:{action[0]}")
        print(f"{w_val:<10.1f} {' '.join(policy_str)}")
    
    # 6. Visualization
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Whittle Indices Comparison
    ax = axes[0, 0]
    x_pos = np.arange(N_states)
    width = 0.35
    
    ax.bar(x_pos - width/2, whittle_indices_binary, width, label='Binary Search', alpha=0.8)
    closed_vals = [W_active] + list(W_passive[:N]) + [np.nan] * (N_states - N - 1)
    ax.bar(x_pos + width/2, closed_vals, width, label='Closed-Form', alpha=0.8)
    
    if PKG_AVAILABLE:
        pkg_vals = [pkg_active_special] + list(pkg_passive[:N])
        ax.bar(x_pos, pkg_vals, width/2, label='Package', alpha=0.8)
    
    ax.set_xlabel('State')
    ax.set_ylabel('Whittle Index')
    ax.set_title('Comparison of Whittle Indices')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(state_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Index values by state type
    ax = axes[0, 1]
    active_idx = [i for i, label in enumerate(state_labels) if label != "(0,1)" and '1' in label]
    passive_idx = [i for i, label in enumerate(state_labels) if '0' in label]
    
    ax.plot(active_idx, whittle_indices_binary[active_idx], 'o-', label='Active States', 
            markersize=8, linewidth=2)
    ax.plot(passive_idx, whittle_indices_binary[passive_idx], 's-', label='Passive States', 
            markersize=8, linewidth=2)
    ax.set_xlabel('State Index')
    ax.set_ylabel('Whittle Index')
    ax.set_title('Indices by State Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Index difference between active and passive
    ax = axes[0, 2]
    if N_states >= 2:
        diff_passive_active = []
        state_labels_short = []
        for x in range(min(N, N_states-1)):
            active_state = f"({x},1)" if x == 0 else f"({x},1)"
            passive_state = f"({x},0)"
            # Find indices
            active_idx = state_labels.index(active_state) if active_state in state_labels else None
            passive_idx = state_labels.index(passive_state) if passive_state in state_labels else None
            
            if active_idx is not None and passive_idx is not None:
                diff = whittle_indices_binary[active_idx] - whittle_indices_binary[passive_idx]
                diff_passive_active.append(diff)
                state_labels_short.append(x)
        
        bars = ax.bar(state_labels_short, diff_passive_active, alpha=0.7, color='coral')
        ax.axhline(y=-C_switch, color='green', linestyle='--', linewidth=2, 
                  label=f'Theoretical: -{C_switch}')
        ax.set_xlabel('State x')
        ax.set_ylabel('W(x,Active) - W(x,Passive)')
        ax.set_title(f'Index Difference (Expected = -{C_switch})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Policy threshold analysis
    ax = axes[1, 0]
    w_range = np.linspace(-30, 60, 100)
    active_preferred = np.zeros((len(w_range), N_states))
    
    for i, w_val in enumerate(w_range):
        policy = get_optimal_policy(P0, P1, R0, R1, w_val, discount)
        active_preferred[i, :] = [policy[s] for s in range(N_states)]
    
    im = ax.imshow(active_preferred.T, aspect='auto', cmap='RdYlGn', 
                   extent=[w_range[0], w_range[-1], -0.5, N_states-0.5])
    ax.set_xlabel('Intervention Charge w')
    ax.set_ylabel('State')
    ax.set_title('Optimal Action vs w (1=Repair, 0=Degrade)')
    ax.set_yticks(range(N_states))
    ax.set_yticklabels(state_labels)
    plt.colorbar(im, ax=ax)
    
    # Plot 5: Comparison of index calculation methods
    ax = axes[1, 1]
    if PKG_AVAILABLE:
        binary_sorted = np.sort(whittle_indices_binary)
        pkg_sorted = np.sort(pkg_vals[:N_states])
        ax.plot(binary_sorted, pkg_sorted, 'o', alpha=0.7, markersize=8)
        ax.plot([binary_sorted.min(), binary_sorted.max()], 
                [binary_sorted.min(), binary_sorted.max()], 
                'r--', linewidth=2, label='Perfect agreement')
        ax.set_xlabel('Binary Search Indices')
        ax.set_ylabel('Package Indices')
        ax.set_title('Method Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = np.corrcoef(whittle_indices_binary, pkg_vals[:N_states])[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    # Plot 6: Convergence of binary search
    ax = axes[1, 2]
    
    def convergence_demo(state_idx=1):
        w_history = []
        w_min, w_max = -100, 100
        for i in range(50):
            w_mid = (w_min + w_max) / 2
            w_history.append(w_mid)
            
            V = value_iteration(P0, P1, R0, R1, w_mid, discount, max_iter=1000)
            Q_passive = R0[state_idx] + discount * np.dot(P0[state_idx, :], V)
            Q_active = (R1[state_idx] - w_mid) + discount * np.dot(P1[state_idx, :], V)
            
            if Q_passive < Q_active:
                w_min = w_mid
            else:
                w_max = w_mid
        
        return w_history
    
    # Demo convergence for first few states
    for state_idx in range(min(3, N_states)):
        w_history = convergence_demo(state_idx)
        ax.plot(w_history, label=state_labels[state_idx])
    
    ax.set_xlabel('Binary Search Iteration')
    ax.set_ylabel('w Value')
    ax.set_title('Convergence of Binary Search')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('whittle_indices_complete_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Figure saved as 'whittle_indices_complete_analysis.png'")
    
    # 7. Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nWhittle Index Statistics:")
    print(f"  Mean: {np.mean(whittle_indices_binary):.4f}")
    print(f"  Std: {np.std(whittle_indices_binary):.4f}")
    print(f"  Min: {np.min(whittle_indices_binary):.4f}")
    print(f"  Max: {np.max(whittle_indices_binary):.4f}")
    print(f"  Range: {np.max(whittle_indices_binary) - np.min(whittle_indices_binary):.4f}")
    
    # Rank ordering of indices
    ranks = rankdata(whittle_indices_binary)
    print(f"\nIndex Ranking (1=lowest):")
    for i, (label, rank) in enumerate(zip(state_labels, ranks)):
        print(f"  {label}: rank {int(rank)} (index = {whittle_indices_binary[i]:.4f})")
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    plt.show()