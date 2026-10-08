import numpy as np
import pandas as pd
import markovianbandit as bandit


def generate_survival_probs(N):
    """
    Generates N+1 survival probabilities such that p0 >= p1 >= ... >= pN = 0.
    Ensures failure probability (1-px) is increasing (and convex).
    """
    # Sample 7 values from U[0, 1] and sort them descending
    samples = np.sort(np.random.uniform(0, 1, N))[::-1]
    # Append 0 at the end as per pN = 0
    p = np.append(samples, 0)
    return p

def create_model(N, p, Cm, Km, c_switch):
    """
    N: Max state index (e.g., N=5 means 6 states 0..5)
    p: list of survival probabilities of length N+1
    """
    num_states = (N + 1) * 2
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)
    
    # Active/Passive offset
    # Indices 0 to N: Passive family (0,0) to (N,0)
    # Indices N+1 to 2N+1: Active family (0,1) to (N,1)
    
    for x in range(N + 1):
        idx_P = x           # Passive: 0 to N
        idx_A = x + (N + 1) # Active: N+1 to 2N+1
        
        # --- Passive Action Dynamics (P0) ---
        # 1. Catastrophic Breakdown (1-px) -> leads to (0,0), index 0
        P0[idx_P, 0] += (1 - p[x])
        P0[idx_A, 0] += (1 - p[x])
        
        # 2. Deterioration (px) -> leads to (x+1, 0)
        if x < N:
            P0[idx_P, x + 1] = p[x]
            P0[idx_A, x + 1] = p[x]
        else:
            # At last state, treat deterioration as failure (or stay)
            # Rule: Last state only has catastrophic failure
            P0[idx_P, 0] += p[x]
            P0[idx_A, 0] += p[x]
            
        # --- Active Action Dynamics (P1) ---
        # Reset to (0,1) -> index N+1
        P1[idx_P, N + 1] = 1.0
        P1[idx_A, N + 1] = 1.0
        
        # --- Reward Vectors ---
        R0[idx_P] = R0[idx_A] = -Km * (1 - p[x])   # -Km
        R1[idx_P] = -(Cm + c_switch)
        R1[idx_A] = -Cm
        
    return P0, P1, R0, R1

# ----------------------------------------------------------------------
# NEW FUNCTIONS FOR AVERAGE COST INDICES (COROLLARY 2)
# ----------------------------------------------------------------------

def compute_T_k(p):
    """
    Compute Ť(k) = Σ_{l=0}^{k-1} Π_{j=0}^{l} p_j for all k
    
    Args:
        p: list of survival probabilities [p0, p1, ..., pN]
        
    Returns:
        T: list of Ť(k) for k=0 to k=N+1 (len = N+2)
    """
    N = len(p)
    T = np.zeros(N + 1)  # T[0], T[1], ..., T[N]
    
    # Base case: Ť(0) = 0
    T[0] = 0.0
    
    # Compute recursively
    for k in range(1, N + 1):
        # Compute product from j=0 to k-1
        product = 1.0
        for j in range(k):
            product *= p[j]
        T[k] = T[k-1] + product
    
    return T

def compute_average_cost_indices(p, Cm, Km):
    """
    Compute average cost Whittle indices using Corollary 2 formula:
    Ŵ(k) = K * [1 - p_k / (Ť(k) - p_k * Ť(k-1))] - C
    
    Args:
        p: list of survival probabilities [p0, p1, ..., pN]
        Cm: intervention/maintenance cost
        Km: breakdown cost
        
    Returns:
        W_avg: list of average cost indices for states 0..N
    """
    N = len(p) - 1  # Last state index
    T = compute_T_k(p)
    
    W_avg = np.zeros(N + 1)
    
    for k in range(N + 1):
        p_k = p[k]
        
        if k == 0:
            # Special case: For k=0, the formula simplifies to K*(1-p0) - C
            # This comes from the limit of the general formula as k->0
            W_avg[k] = Km * (1 - p_k) - Cm
        else:
            # General formula from Corollary 2
            T_k = T[k]      # Ť(k)
            T_km1 = T[k-1]  # Ť(k-1)
            
            denominator = T_k - p_k * T_km1
            
            # Handle potential division by zero (when p_k is very small)
            if abs(denominator) < 1e-12:
                # When denominator is 0, index approaches K - C
                W_avg[k] = Km - Cm
            else:
                W_avg[k] = Km * (1 - p_k / denominator) - Cm
    
    return W_avg

def compute_theoretical_indices(p, Cm, Km, beta):
    """
    Computes theoretical Whittle indices based on Glazebrook et al. (2005)
    Theorem 2: W(k) = Km * [ (1 - G(k+1) - p_k * (1 - beta*G(k))) / 
                            (1 - G(k+1) - beta * p_k * (1 - G(k))) ] - Cm
    """
    N = len(p) - 1
    G = np.zeros(N + 2)
    H = np.zeros(N + 2)
    
    # Initialize G and H
    G[0] = 0
    H[0] = 1 # Product over empty set is 1
    
    # Compute G(k) and H(k) iteratively using the identity (25)
    for k in range(N + 1):
        H[k+1] = H[k] * beta * p[k]
        G[k+1] = G[k] + (1 - p[k]) * H[k]
        
    indices = []
    for k in range(N + 1):
        # Formula (26)
        num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
        den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
        W = Km * (num / den) - Cm
        indices.append(W)
    return np.array(indices)

# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------

# --- Configuration ---
N = 7 
beta = 0.9
p = generate_survival_probs(N)
print("Survival Probabilities (p_x):", np.round(p, 3))
print("Failure Probabilities (1-p_x):", np.round(1-p, 3))
p[-1] = 0 # Ensure last state is 0 survival
Cm, Km, c_switch = 5, 500, 0

P0, P1, R0, R1 = create_model(N, p, Cm, Km, c_switch)

# Labels for pretty printing
labels = [f"({x},0)" for x in range(N + 1)] + [f"({x},1)" for x in range(N + 1)]

print("\n--- Transition Matrix P0 (Passive Action) ---")
print(pd.DataFrame(P0, index=labels, columns=labels))
print("\n--- Transition Matrix P1 (Active Action) ---")
print(pd.DataFrame(P1, index=labels, columns=labels))
print("\n--- Reward Vector R0 (Passive Costs) ---")
print(pd.DataFrame(R0, index=labels, columns=["Value"]))
print("\n--- Reward Vector R1 (Active Costs) ---")
print(pd.DataFrame(R1, index=labels, columns=["Value"]))

# Solve
model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
discounted_indices = model.whittle_indices(discount=beta)
average_indices = model.whittle_indices(discount=0.0)

# Compute theoretical indices
theoretical_W = compute_theoretical_indices(p, Cm, Km, beta)

# ----------------------------------------------------------------------
# NEW: Compute average cost indices using Corollary 2
# ----------------------------------------------------------------------

# Compute T(k) values for verification
T_values = compute_T_k(p)
print("\n--- Computed Ť(k) values (Corollary 2) ---")
for k in range(N + 1):
    print(f"Ť({k}) = {T_values[k]:.6f}")

# Compute average cost indices using Corollary 2
average_cost_theoretical = compute_average_cost_indices(p, Cm, Km)

print("\n--- Comparison Table (All Indices) ---")
print(f"{'State':<10} | {'p_k':<10} | {'Discounted (β=0.9)':<20} | {'Average (Package)':<20} | {'Average (Theory)':<20}")
print("-" * 85)
for i in range(N + 1):
    print(f"{i:<10} | {p[i]:<10.4f} | {discounted_indices[i]:<20.4f} | {average_indices[i]:<20.4f} | {average_cost_theoretical[i]:<20.4f}")

# ----------------------------------------------------------------------
# Additional Analysis: Compare with package average indices
# ----------------------------------------------------------------------

print("\n--- Difference Analysis: Package vs Theoretical Average Indices ---")
print(f"{'State':<10} | {'Package Avg':<15} | {'Theory Avg':<15} | {'Absolute Diff':<15} | {'Relative Diff (%)':<15}")
print("-" * 70)
for i in range(N + 1):
    abs_diff = abs(average_indices[i] - average_cost_theoretical[i])
    rel_diff = abs_diff / max(abs(average_indices[i]), 1e-10) * 100
    print(f"{i:<10} | {average_indices[i]:<15.4f} | {average_cost_theoretical[i]:<15.4f} | {abs_diff:<15.6f} | {rel_diff:<15.4f}")

# ----------------------------------------------------------------------
# Verification: Compute indices for β close to 1 to see convergence
# ----------------------------------------------------------------------

print("\n--- Convergence Check: Discounted indices as β → 1 ---")
beta_values = [0.9, 0.95, 0.99, 0.999]
for beta_val in beta_values:
    indices_beta = compute_theoretical_indices(p, Cm, Km, beta_val)
    # Take average indices for comparison
    avg_idx = np.mean(np.abs(indices_beta - average_cost_theoretical))
    print(f"β = {beta_val:.3f}: Average absolute difference from average cost indices = {avg_idx:.6f}")

# ----------------------------------------------------------------------
# Special State Analysis
# ----------------------------------------------------------------------

print("\n--- Special State Analysis ---")
print("State 0:")
print(f"  p_0 = {p[0]:.4f}")
print(f"  Ŵ(0) formula: K*(1-p_0) - C = {Km}*(1-{p[0]:.4f}) - {Cm} = {Km*(1-p[0]) - Cm:.4f}")
print(f"  Actual computed: {average_cost_theoretical[0]:.4f}")

print("\nState N (last state):")
print(f"  p_{N} = {p[N]:.4f}")
print(f"  Ŵ(N) formula: K - C = {Km} - {Cm} = {Km-Cm:.4f}")
print(f"  Actual computed: {average_cost_theoretical[N]:.4f}")

# ----------------------------------------------------------------------
# Plot convergence if matplotlib is available
# ----------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    
    # Plot all three indices
    states = list(range(N + 1))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(states, discounted_indices[:N+1], 'bo-', label='Discounted (β=0.9)', markersize=8)
    plt.plot(states, average_indices[:N+1], 'rs-', label='Average (Package)', markersize=8)
    plt.plot(states, average_cost_theoretical, 'g^--', label='Average (Theory)', markersize=8)
    plt.xlabel('State (k)')
    plt.ylabel('Whittle Index')
    plt.title('Whittle Indices Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(states, p, 'ko-', label='Survival Probability p_k')
    plt.xlabel('State (k)')
    plt.ylabel('p_k')
    plt.title('Survival Probabilities')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("\nMatplotlib not available. Skipping plots.")