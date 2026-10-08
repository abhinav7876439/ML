import numpy as np
import markovianbandit as bandit

def generate_ordered_probs(N=3):
    # N=3 means 4 states (0,1,2,3).
    # We need 3 random values, sorted descending, then append 0.
    samples = np.sort(np.random.uniform(0.2, 0.9, N))[::-1]
    return np.append(samples, 0)

def print_formatted_matrix(name, matrix):
    size = matrix.shape[0]
    print(f"\n--- {name} ({size}x{size}) ---")
    print(np.array_str(matrix, precision=2, suppress_small=True))

def create_full_model(N, p, Cm, Km, c_switch):
    num_states = (N + 1) * 2 # 8 states total
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)
    
    # Active indices: 0..3 | Passive indices: 4..7
    for x in range(N + 1):
        idx_A = x          
        idx_P = x + (N + 1) 
        
        # --- P0 (Passive) ---
        # Failure leads to State 0 in Passive Family (N+1 = index 4)
        P0[idx_A, N + 1] = (1 - p[x])
        P0[idx_P, N + 1] = (1 - p[x])
        
        if x < N:
            P0[idx_A, idx_P + 1] = p[x] # Deteriorate
            P0[idx_P, idx_P + 1] = p[x]
        else:
            P0[idx_A, N + 1] += p[x]    # Last state failure
            P0[idx_P, N + 1] += p[x]
            
        # --- P1 (Active) ---
        P1[idx_A, 0] = 1.0              # Reset to index 0
        P1[idx_P, 0] = 1.0
        
        # --- Costs ---
        R0[idx_A] = R0[idx_P] = -Km * (1 - p[x])
        R1[idx_A] = -Cm
        R1[idx_P] = -(Cm + c_switch)
        
    return P0, P1, R0, R1

# Parameters (N=3 implies 4 states)
N = 3
p = generate_ordered_probs(N)
Cm, Km, c_switch = 5, 50, 10

P0, P1, R0, R1 = create_full_model(N, p, Cm, Km, c_switch)

# Print Output
print_formatted_matrix("Transition Matrix P0 (Passive)", P0)
print_formatted_matrix("Transition Matrix P1 (Active)", P1)
print("\n--- Reward Vector R0 (Passive) ---")
print(np.round(R0, 1))
print("\n--- Reward Vector R1 (Active) ---")
print(np.round(R1, 1))

# Final Calculation
model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
indices = model.whittle_indices(discount=0.9)

print("\nCalculated Whittle Indices for all 8 states (1-4 Active, 5-8 Passive):")
print(np.round(indices, 3))




####################################################


import numpy as np
import pandas as pd
import markovianbandit as bandit

def print_pretty(name, data, is_matrix=True):
    print(f"\n{'='*20} {name} {'='*20}")
    labels = [f"({x},0)" for x in range(4)] + [f"({x},1)" for x in range(4)]
    if is_matrix:
        df = pd.DataFrame(data, index=labels, columns=labels)
        print(df)
    else:
        df = pd.DataFrame(data, index=labels, columns=["Value"])
        print(df.T)

# Parameters
N, Cm, Km, c_switch = 3, 5, 50, 10
p = [0.9, 0.7, 0.5, 0.3]

# Create Model
# 1-4: Passive (0,0)-(3,0) | 5-8: Active (0,1)-(3,1)
P0 = np.zeros((8, 8))
P1 = np.zeros((8, 8))
R0 = np.zeros(8)
R1 = np.zeros(8)

for x in range(N + 1):
    idx_P = x          # Passive: 0-3
    idx_A = x + 4      # Active: 4-7
    
    # --- Passive Action Dynamics ---
    # Failure -> State 1 (0,0) index 0
    P0[idx_P, 0] = (1 - p[x])
    P0[idx_A, 0] = (1 - p[x])
    # Deterioration -> State idx_P + 1 (if x < 3)
    if x < N:
        P0[idx_P, idx_P + 1] = p[x]
        P0[idx_A, idx_P + 1] = p[x]
    else:
        # Last state: failure only
        P0[idx_P, 0] += p[x]
        P0[idx_A, 0] += p[x]
        
    # --- Active Action Dynamics ---
    # Reset to State 5 (0,1) index 4
    P1[idx_P, 4] = 1.0
    P1[idx_A, 4] = 1.0
    
    # Costs
    R0[idx_P] = R0[idx_A] = -Km * (1 - p[x])
    R1[idx_P] = -(Cm + c_switch)
    R1[idx_A] = -Cm

# Display Matrices
print_pretty("Transition Matrix P0 (Passive Action)", P0)
print_pretty("Transition Matrix P1 (Active Action)", P1)
print_pretty("Reward Vector R0 (Passive Costs)", R0, is_matrix=False)
print_pretty("Reward Vector R1 (Active Costs)", R1, is_matrix=False)

# Final calculation
model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
indices = model.whittle_indices(discount=0.9)
print("\n--- Final Whittle Indices ---")
for i, val in enumerate(indices):
    label = f"State {i+1} {'(0,0)' if i<4 else '(1,1)'}"
    print(f"{label}: {val:.4f}")



#######################################################






#######################################################

import numpy as np
import markovianbandit as bandit

def solve_machine_repairman(N, p, Cm, Km, c_switch, beta=0.9):
    """
    N: Max state (3)
    p: list of survival probabilities [p0, p1, p2, p3]
    """
    num_states = (N + 1) * 2 # 8 states (1-4 Active, 5-8 Passive)
    
    # Initialize P0 (Passive) and P1 (Active)
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    
    # Costs
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)
    
    # Mapping indices: 
    # Active: 0-3 (States 1-4) | Passive: 4-7 (States 5-8)
    for x in range(N + 1):
        idx_A = x          # Active family (x, 1)
        idx_P = x + (N + 1) # Passive family (x, 0)
        
        # --- P0 (Passive Action b=0) ---
        # 1. Failure leads to (0,0) -> index 4 (State 5)
        P0[idx_A, 4] = (1 - p[x])
        P0[idx_P, 4] = (1 - p[x])
        
        # 2. Deterioration leads to (x+1, 0) -> index 5+x
        if x < N:
            P0[idx_A, idx_P + 1] = p[x]
            P0[idx_P, idx_P + 1] = p[x]
        else:
            # Last state logic: catastrophic failure with prob 1
            P0[idx_A, 4] += p[x]
            P0[idx_P, 4] += p[x]
            
        # --- P1 (Active Action a=1) ---
        # Deterministic: move to (0,1) -> index 0 (State 1)
        P1[idx_A, 0] = 1.0
        P1[idx_P, 0] = 1.0
        
        # --- Costs ---
        # Passive cost: Expected failure cost K * (1 - px)
        R0[idx_A] = R0[idx_P] = -Km * (1 - p[x])
        # Active cost: Repair + Switch cost (only if coming from passive)
        R1[idx_A] = -Cm
        R1[idx_P] = -(Cm + c_switch)
        
    # --- Solve ---
    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    indices = model.whittle_indices(discount=beta)
    
    return indices

# --- Simulation Parameters ---
N = 3
p = [0.9, 0.7, 0.5, 0.3] # Decreasing survival probabilities
Cm, Km, c_switch = 5, 500, 0

indices = solve_machine_repairman(N, p, Cm, Km, c_switch)

# Display results mapping indices 1-8
print(f"{'State':<10} | {'Family':<10} | {'Whittle Index':<15}")
print("-" * 40)
for i in range(8):
    family = "Active" if i < 4 else "Passive"
    print(f"{i+1:<10} | {family:<10} | {indices[i]:.4f}")