import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: Define System Parameters (4-State Machine)
# ============================================================================

N_STATES = 4
DISCOUNT = 0.95
C_REPAIR = 10.0
K_FAILURE = 100.0
C_SWITCH = 5.0
p_survive = np.array([0.9, 0.7, 0.4, 0.1])

print("=" * 70)
print("SYSTEM PARAMETERS")
print("=" * 70)
print(f"States: {N_STATES}")
print(f"Discount factor β: {DISCOUNT}")
print(f"Repair cost C: {C_REPAIR}")
print(f"Failure cost K: {K_FAILURE}")
print(f"Switching cost c_switch: {C_SWITCH}")
print(f"Survival probabilities p: {p_survive}")
print()

# ============================================================================
# STEP 2: Extended State Space
# ============================================================================

def state_to_idx(x, a_prev):
    return x * 2 + a_prev

print(f"State to Index Mapping:")
for x in range(N_STATES):
    for a_prev in [0, 1]:
        idx = state_to_idx(x, a_prev)
        a_label = "Active" if a_prev == 1 else "Passive"
        print(f"State (x={x}, a_prev={a_label}) -> Index {idx}")

def idx_to_state(idx):
    return idx // 2, idx % 2
print(f"\nIndex to State Mapping:")
for idx in range(N_STATES * 2):
    x, a_prev = idx_to_state(idx)
    a_label = "Active" if a_prev == 1 else "Passive"
    print(f"Index {idx} -> State (x={x}, a_prev={a_label})")

N_EXTENDED_STATES = N_STATES * 2
print(f"Extended state space size: {N_EXTENDED_STATES}")
print()

# ============================================================================
# STEP 3: Build Transition Matrices
# ============================================================================

P0 = np.zeros((N_EXTENDED_STATES, N_EXTENDED_STATES))
P1 = np.zeros((N_EXTENDED_STATES, N_EXTENDED_STATES))
R0 = np.zeros(N_EXTENDED_STATES)
R1 = np.zeros(N_EXTENDED_STATES)

for idx in range(N_EXTENDED_STATES):
    x, a_prev = idx_to_state(idx)
    
    # PASSIVE ACTION
    if x < N_STATES - 1:
        next_idx_survive = state_to_idx(x + 1, 0)
        P0[idx, next_idx_survive] = p_survive[x]
    
    next_idx_fail = state_to_idx(0, 0)
    P0[idx, next_idx_fail] += (1 - p_survive[x])
    R0[idx] = -K_FAILURE * (1 - p_survive[x])
    
    # ACTIVE ACTION
    next_idx_repair = state_to_idx(0, 1)
    P1[idx, next_idx_repair] = 1.0
    
    if a_prev == 1:
        R1[idx] = -C_REPAIR
    else:
        R1[idx] = -(C_REPAIR + C_SWITCH)

print("✓ Transition Matrices Constructed")
print()
print("Sample Transition Probabilities (Passive):")
for x in range(N_STATES):   
    idx_passive = state_to_idx(x, 0)
    print(f"From (x={x}, Passive): {P0[idx_passive, :]}")
print("\nSample Transition Probabilities (Active):")
for x in range(N_STATES):   
    idx_active = state_to_idx(x, 1)
    print(f"From (x={x}, Active): {P1[idx_active, :]}")
print()
print("Sample Rewards:")        
for x in range(N_STATES):
    idx_passive = state_to_idx(x, 0)
    idx_active = state_to_idx(x, 1)
    print(f"State (x={x}): R0={R0[idx_passive]:.2f}, R1={R1[idx_active]:.2f}")  
print() 




# ============================================================================
# STEP 4: Closed-Form Whittle Indices (Value Iteration + Binary Search)
# ============================================================================

def value_iteration(P0, P1, R0, R1, w_val, discount, max_iter=1000):
    """Compute value function for given intervention charge w"""
    V = np.zeros(N_EXTENDED_STATES)
    
    for iteration in range(max_iter):
        V_old = V.copy()
        
        for idx in range(N_EXTENDED_STATES):
            Q_passive = R0[idx] + discount * np.dot(P0[idx, :], V_old)
            Q_active = (R1[idx] - w_val) + discount * np.dot(P1[idx, :], V_old)
            V[idx] = min(Q_passive, Q_active)
        
        if np.max(np.abs(V - V_old)) < 1e-8:
            break
    
    return V

def compute_whittle_indices(P0, P1, R0, R1, discount):
    """Compute Whittle indices using binary search"""
    W_indices = np.zeros(N_EXTENDED_STATES)
    
    for state_idx in range(N_EXTENDED_STATES):
        # Binary search for indifference point
        w_low, w_high = -50.0, 50.0
        
        for _ in range(50):  # Binary search iterations
            w_mid = (w_low + w_high) / 2
            V = value_iteration(P0, P1, R0, R1, w_mid, discount)
            
            # Q-values for this state
            Q_passive = R0[state_idx] + discount * np.dot(P0[state_idx, :], V)
            Q_active = (R1[state_idx] - w_mid) + discount * np.dot(P1[state_idx, :], V)
            
            # Adjust binary search bounds
            if Q_passive < Q_active:  # Passive is better
                w_low = w_mid
            else:  # Active is better
                w_high = w_mid
        
        W_indices[state_idx] = (w_low + w_high) / 2
    
    return W_indices

print("=" * 70)
print("COMPUTING WHITTLE INDICES")
print("=" * 70)
print("Computing via binary search on intervention charge w...")
print()

whittle_indices = compute_whittle_indices(P0, P1, R0, R1, DISCOUNT)

print("=" * 70)
print("WHITTLE INDICES")
print("=" * 70)
print(f"{'State':<20} {'Index Value':<15}")
print("-" * 35)

for idx in range(N_EXTENDED_STATES):
    x, a_prev = idx_to_state(idx)
    a_label = "Active " if a_prev == 1 else "Passive"
    state_label = f"({x}, {a_label})"
    print(f"{state_label:<20} {whittle_indices[idx]:>14.6f}")

print()

# ============================================================================
# STEP 5: Index Structure Analysis
# ============================================================================

print("=" * 70)
print("INDEX STRUCTURE ANALYSIS")
print("=" * 70)

print("\nActive vs Passive Comparison (same x):")
print(f"{'State x':<10} {'Active (x,1)':<15} {'Passive (x,0)':<15} {'Difference':<15}")
print("-" * 55)

for x in range(N_STATES):
    idx_active = state_to_idx(x, 1)
    idx_passive = state_to_idx(x, 0)
    diff = whittle_indices[idx_active] - whittle_indices[idx_passive]
    print(f"{x:<10} {whittle_indices[idx_active]:<15.6f} {whittle_indices[idx_passive]:<15.6f} "
          f"{diff:<15.6f}")

print()
print(f"Expected difference: ≈ {-C_SWITCH:.6f} (negative switching cost)")
print()

# ============================================================================
# STEP 6: Optimal Policy Analysis
# ============================================================================

def get_optimal_policy(P0, P1, R0, R1, w_val, discount):
    """Get optimal policy for given intervention charge"""
    V = value_iteration(P0, P1, R0, R1, w_val, discount)
    
    policy = {}
    for s in range(N_EXTENDED_STATES):
        Q_passive = R0[s] + discount * np.dot(P0[s, :], V)
        Q_active = (R1[s] - w_val) + discount * np.dot(P1[s, :], V)
        policy[s] = "Repair" if Q_active <= Q_passive else "Degrade"
    
    return policy

print("=" * 70)
print("OPTIMAL POLICY AT DIFFERENT INTERVENTION CHARGES")
print("=" * 70)

w_test_values = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]

print(f"\n{'w':<10} {'Active States':<35} {'Passive States':<35}")
print("-" * 80)

for w_test in w_test_values:
    policy = get_optimal_policy(P0, P1, R0, R1, w_test, DISCOUNT)
    
    active_policy_str = ", ".join([f"x{x}:{policy[state_to_idx(x, 1)][0]}" for x in range(N_STATES)])
    passive_policy_str = ", ".join([f"x{x}:{policy[state_to_idx(x, 0)][0]}" for x in range(N_STATES)])
    
    print(f"{w_test:<10.1f} {active_policy_str:<35} {passive_policy_str:<35}")

print()
print("Legend: R=Repair, D=Degrade")
print()

# ============================================================================
# STEP 7: Effect of Switching Cost
# ============================================================================

print("=" * 70)
print("EFFECT OF SWITCHING COST ON INDICES")
print("=" * 70)

c_switch_values = [0.0, 2.5, 5.0, 7.5, 10.0]
indices_by_cswitch = {}

for c_switch_test in c_switch_values:
    # Rebuild cost vectors
    R0_test = np.zeros(N_EXTENDED_STATES)
    R1_test = np.zeros(N_EXTENDED_STATES)
    
    for idx in range(N_EXTENDED_STATES):
        x, a_prev = idx_to_state(idx)
        R0_test[idx] = -K_FAILURE * (1 - p_survive[x])
        if a_prev == 1:
            R1_test[idx] = -C_REPAIR
        else:
            R1_test[idx] = -(C_REPAIR + c_switch_test)
    
    indices_by_cswitch[c_switch_test] = compute_whittle_indices(P0, P1, R0_test, R1_test, DISCOUNT)

print(f"{'c_switch':<12} {'W(0,Active)':<15} {'W(0,Passive)':<15} {'Difference':<15}")
print("-" * 57)

for c_switch_test in c_switch_values:
    indices = indices_by_cswitch[c_switch_test]
    idx_active = state_to_idx(0, 1)
    idx_passive = state_to_idx(0, 0)
    diff = indices[idx_active] - indices[idx_passive]
    
    print(f"{c_switch_test:<12.1f} {indices[idx_active]:<15.6f} {indices[idx_passive]:<15.6f} "
          f"{diff:<15.6f}")

print()
print(f"Verification: Differences should equal -c_switch values")
print()

# ============================================================================
# STEP 8: Visualization
# ============================================================================

print("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Indices by State
ax = axes[0, 0]
x_pos = np.arange(N_EXTENDED_STATES)
colors = ['blue' if i % 2 == 1 else 'orange' for i in range(N_EXTENDED_STATES)]
ax.bar(x_pos, whittle_indices, color=colors, alpha=0.7)
ax.set_xlabel('Extended State Index')
ax.set_ylabel('Whittle Index')
ax.set_title('Whittle Indices by State')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{idx_to_state(i)}" for i in range(N_EXTENDED_STATES)], fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.legend(['Active', 'Passive'], loc='upper right')

# Plot 2: Active vs Passive Index Difference
ax = axes[0, 1]
diffs = []
for x in range(N_STATES):
    idx_active = state_to_idx(x, 1)
    idx_passive = state_to_idx(x, 0)
    diffs.append(whittle_indices[idx_active] - whittle_indices[idx_passive])

ax.bar(range(N_STATES), diffs, color='red', alpha=0.7)
ax.axhline(y=-C_SWITCH, color='green', linestyle='--', linewidth=2, label=f'Expected: -{C_SWITCH}')
ax.set_xlabel('State x')
ax.set_ylabel('W(x,Active) - W(x,Passive)')
ax.set_title('Index Difference by State')
ax.set_xticks(range(N_STATES))
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: Effect of Switching Cost
ax = axes[1, 0]
c_switch_array = np.array(c_switch_values)
idx_0_active = state_to_idx(0, 1)
idx_0_passive = state_to_idx(0, 0)

indices_0_active = [indices_by_cswitch[cs][idx_0_active] for cs in c_switch_values]
indices_0_passive = [indices_by_cswitch[cs][idx_0_passive] for cs in c_switch_values]

ax.plot(c_switch_array, indices_0_active, 'o-', label='(0, Active)', linewidth=2, markersize=8)
ax.plot(c_switch_array, indices_0_passive, 's-', label='(0, Passive)', linewidth=2, markersize=8)
ax.set_xlabel('Switching Cost $c_{switch}$')
ax.set_ylabel('Whittle Index')
ax.set_title('Effect of Switching Cost on Indices')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Index Difference vs Switching Cost
ax = axes[1, 1]
index_diffs = [indices_by_cswitch[cs][idx_0_active] - indices_by_cswitch[cs][idx_0_passive] 
               for cs in c_switch_values]
ax.plot(c_switch_array, index_diffs, 'D-', color='purple', linewidth=2, markersize=8, 
        label='$W(0,Active) - W(0,Passive)$')
ax.plot(c_switch_array, -c_switch_array, '--', color='green', linewidth=2, label='$-c_{switch}$ (Expected)')
ax.set_xlabel('Switching Cost $c_{switch}$')
ax.set_ylabel('Index Difference')
ax.set_title('Verification: Index Difference vs -$c_{switch}$')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('whittle_indices_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Figure saved as 'whittle_indices_analysis.png'")
print()

print("=" * 70)
print("✓ EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 70)