import numpy as np
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import markovianbandit as bandit
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
# STEP 2: Construct Extended State Space (x, a_prev)
# ============================================================================

def state_to_idx(x, a_prev):
    return x * 2 + a_prev

def idx_to_state(idx):
    return idx // 2, idx % 2

N_EXTENDED_STATES = N_STATES * 2
print(f"Extended state space size: {N_EXTENDED_STATES}")
print()

# ============================================================================
# STEP 3: Build Transition Matrices P0 (Passive) and P1 (Active)
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

print("Transition Matrices Constructed:")
print(f"P0 shape (Passive): {P0.shape}")
print(f"P1 shape (Active):  {P1.shape}")
print()

# ============================================================================
# STEP 4: Compute Whittle Indices via Package
# ============================================================================

print("Computing indices from package (this may take a moment)...")
model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
whittle_indices_package = model.whittle_indices(discount=DISCOUNT)

print("=" * 70)
print("WHITTLE INDICES FROM PACKAGE")
print("=" * 70)
for idx in range(N_EXTENDED_STATES):
    x, a_prev = idx_to_state(idx)
    a_label = "Active" if a_prev == 1 else "Passive"
    print(f"State ({x}, {a_label:7s}): W_{idx} = {whittle_indices_package[idx]:10.6f}")
print()

# ============================================================================
# STEP 5: Closed-Form Computation via Value Iteration
# ============================================================================

def value_iteration_with_intervention_charge(P0, P1, R0, R1, w_val, discount, tol=1e-8, max_iter=5000):
    V = np.zeros(N_EXTENDED_STATES)
    
    for iteration in range(max_iter):
        V_old = V.copy()
        
        for idx in range(N_EXTENDED_STATES):
            Q_passive = R0[idx] + discount * np.dot(P0[idx, :], V_old)
            Q_active = (R1[idx] - w_val) + discount * np.dot(P1[idx, :], V_old)
            V[idx] = min(Q_passive, Q_active)
        
        if np.max(np.abs(V - V_old)) < tol:
            break
    
    return V

def compute_whittle_indices_closed_form(P0, P1, R0, R1, discount, w_min=-20, w_max=50, resolution=500):
    W_indices = np.zeros(N_EXTENDED_STATES)
    w_values = np.linspace(w_min, w_max, resolution)
    
    for idx in range(N_EXTENDED_STATES):
        x, a_prev = idx_to_state(idx)
        advantages = []
        
        for w_val in w_values:
            V = value_iteration_with_intervention_charge(P0, P1, R0, R1, w_val, discount, max_iter=500)
            V_passive = R0[idx] + discount * np.dot(P0[idx, :], V)
            V_active = (R1[idx] - w_val) + discount * np.dot(P1[idx, :], V)
            advantage = V_passive - V_active
            advantages.append(advantage)
        
        advantages = np.array(advantages)
        sign_changes = np.where(np.diff(np.sign(advantages)))[0]
        
        if len(sign_changes) > 0:
            idx_cross = sign_changes[0]
            w1, w2 = w_values[idx_cross], w_values[idx_cross + 1]
            adv1, adv2 = advantages[idx_cross], advantages[idx_cross + 1]
            
            if abs(adv2 - adv1) > 1e-12:
                W_indices[idx] = w1 + (w2 - w1) * (0 - adv1) / (adv2 - adv1)
            else:
                W_indices[idx] = w1
        else:
            W_indices[idx] = w_max if advantages[-1] > 0 else w_min
    
    return W_indices

print("=" * 70)
print("COMPUTING WHITTLE INDICES (CLOSED-FORM)")
print("=" * 70)
print("This uses the definition: W*(x) = value of w where active and passive are indifferent")
print()

whittle_indices_closed_form = compute_whittle_indices_closed_form(P0, P1, R0, R1, DISCOUNT)

print()
print("=" * 70)
print("WHITTLE INDICES FROM CLOSED-FORM")
print("=" * 70)
for idx in range(N_EXTENDED_STATES):
    x, a_prev = idx_to_state(idx)
    a_label = "Active" if a_prev == 1 else "Passive"
    print(f"State ({x}, {a_label:7s}): W_{idx} = {whittle_indices_closed_form[idx]:10.6f}")
print()

# ============================================================================
# STEP 6: Comparison and Error Analysis
# ============================================================================

print("=" * 70)
print("COMPARISON: Package vs Closed-Form")
print("=" * 70)
print(f"{'State':<15} {'Package':<15} {'Closed-Form':<15} {'Difference':<15} {'Rel. Error %':<15}")
print("-" * 75)

differences = whittle_indices_package - whittle_indices_closed_form
rel_errors = 100 * np.abs(differences) / (np.abs(whittle_indices_package) + 1e-10)

for idx in range(N_EXTENDED_STATES):
    x, a_prev = idx_to_state(idx)
    a_label = "Active" if a_prev == 1 else "Passive"
    state_label = f"({x}, {a_label})"
    print(f"{state_label:<15} {whittle_indices_package[idx]:<15.6f} {whittle_indices_closed_form[idx]:<15.6f} "
          f"{differences[idx]:<15.6e} {rel_errors[idx]:<15.2f}")

print()
print(f"Max absolute difference: {np.max(np.abs(differences)):.6e}")
print(f"Max relative error: {np.max(rel_errors):.2f}%")
print()

# ============================================================================
# STEP 7: Policy Structure Analysis
# ============================================================================

print("=" * 70)
print("POLICY STRUCTURE ANALYSIS")
print("=" * 70)

w_test_values = np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])

print(f"\nOptimal Policy Thresholds for different intervention charges w:")
print(f"{'w':<10} {'Active States':<30} {'Passive States':<30}")
print("-" * 70)

for w_test in w_test_values:
    V_test = value_iteration_with_intervention_charge(P0, P1, R0, R1, w_test, DISCOUNT)
    
    active_policy = []
    passive_policy = []
    
    for x in range(N_STATES):
        idx_active = state_to_idx(x, 1)
        Q_passive_a = R0[idx_active] + DISCOUNT * np.dot(P0[idx_active, :], V_test)
        Q_active_a = (R1[idx_active] - w_test) + DISCOUNT * np.dot(P1[idx_active, :], V_test)
        action_active = "R" if Q_active_a <= Q_passive_a else "D"
        active_policy.append(f"x{x}:{action_active}")
        
        idx_passive = state_to_idx(x, 0)
        Q_passive_p = R0[idx_passive] + DISCOUNT * np.dot(P0[idx_passive, :], V_test)
        Q_active_p = (R1[idx_passive] - w_test) + DISCOUNT * np.dot(P1[idx_passive, :], V_test)
        action_passive = "R" if Q_active_p <= Q_passive_p else "D"
        passive_policy.append(f"x{x}:{action_passive}")
    
    active_str = ", ".join(active_policy)
    passive_str = ", ".join(passive_policy)
    print(f"{w_test:<10.1f} {active_str:<30} {passive_str:<30}")

print()
print("Legend: R=Repair, D=Degrade")
print()

# ============================================================================
# STEP 8: Effect of Switching Cost
# ============================================================================

print("=" * 70)
print("EFFECT OF SWITCHING COST")
print("=" * 70)

c_switch_values = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
index_differences_by_cswitch = {}

for c_switch_test in c_switch_values:
    R0_test = np.zeros(N_EXTENDED_STATES)
    R1_test = np.zeros(N_EXTENDED_STATES)
    
    for idx in range(N_EXTENDED_STATES):
        x, a_prev = idx_to_state(idx)
        R0_test[idx] = -K_FAILURE * (1 - p_survive[x])
        if a_prev == 1:
            R1_test[idx] = -C_REPAIR
        else:
            R1_test[idx] = -(C_REPAIR + c_switch_test)
    
    model_test = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0_test, R1_test)
    indices_test = model_test.whittle_indices(discount=DISCOUNT)
    index_differences_by_cswitch[c_switch_test] = indices_test

print(f"{'c_switch':<15} {'W(0,Active)':<15} {'W(0,Passive)':<15} {'Difference':<15}")
print("-" * 60)

for c_switch_test in c_switch_values:
    indices = index_differences_by_cswitch[c_switch_test]
    idx_active_0 = state_to_idx(0, 1)
    idx_passive_0 = state_to_idx(0, 0)
    diff = indices[idx_active_0] - indices[idx_passive_0]
    print(f"{c_switch_test:<15.1f} {indices[idx_active_0]:<15.6f} {indices[idx_passive_0]:<15.6f} "
          f"{diff:<15.6f}")

print()
print(f"Expected: Difference ≈ -{C_SWITCH:.1f} (due to switching cost penalty)")
print()

# ============================================================================
# STEP 9: Visualization
# ============================================================================

print("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Package vs Closed-Form Indices
ax = axes[0, 0]
x_pos = np.arange(N_EXTENDED_STATES)
width = 0.35
ax.bar(x_pos - width/2, whittle_indices_package, width, label='Package', alpha=0.8)
ax.bar(x_pos + width/2, whittle_indices_closed_form, width, label='Closed-Form', alpha=0.8)
ax.set_xlabel('Extended State Index')
ax.set_ylabel('Whittle Index')
ax.set_title('Comparison: Package vs Closed-Form Indices')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{idx_to_state(i)}" for i in range(N_EXTENDED_STATES)], fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Absolute Differences
ax = axes[0, 1]
ax.bar(x_pos, np.abs(differences), color='red', alpha=0.7)
ax.set_xlabel('Extended State Index')
ax.set_ylabel('Absolute Difference')
ax.set_title('Absolute Difference |Package - Closed-Form|')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{idx_to_state(i)}" for i in range(N_EXTENDED_STATES)], fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Plot 3: Effect of Switching Cost
ax = axes[1, 0]
c_switch_array = np.array(list(c_switch_values))
idx_0_active = state_to_idx(0, 1)
idx_0_passive = state_to_idx(0, 0)

indices_0_active = [index_differences_by_cswitch[cs][idx_0_active] for cs in c_switch_values]
indices_0_passive = [index_differences_by_cswitch[cs][idx_0_passive] for cs in c_switch_values]

ax.plot(c_switch_array, indices_0_active, 'o-', label='(0, Active)', linewidth=2, markersize=8)
ax.plot(c_switch_array, indices_0_passive, 's-', label='(0, Passive)', linewidth=2, markersize=8)
ax.set_xlabel('Switching Cost $c_{switch}$')
ax.set_ylabel('Whittle Index')
ax.set_title('Effect of Switching Cost on Indices')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Index Differences vs Switching Cost
ax = axes[1, 1]
index_diffs = [index_differences_by_cswitch[cs][idx_0_active] - 
               index_differences_by_cswitch[cs][idx_0_passive] 
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