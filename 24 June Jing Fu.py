import numpy as np

# =============================================================================
# 1. Problem parameters
# =============================================================================
I = 500          # number of arms
K = 50           # number of arms that can be active at any step

# States: 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4
num_states = 4

# Transition matrices (active = P1, passive = P0)
P1 = np.array([[0.5, 0.5, 0.0, 0.0],
               [0.0, 0.5, 0.5, 0.0],
               [0.0, 0.0, 0.5, 0.5],
               [0.5, 0.0, 0.0, 0.5]])

P0 = np.array([[0.5, 0.0, 0.0, 0.5],
               [0.5, 0.5, 0.0, 0.0],
               [0.0, 0.5, 0.5, 0.0],
               [0.0, 0.0, 0.5, 0.5]])

# Rewards: R(state, action) – here action independent
# state 0 (orig 1): -1, state 1 (orig 2): 0, state 2 (orig 3): 0, state 3 (orig 4): +1
R = np.array([-1.0, 0.0, 0.0, 1.0])

# Whittle indices (given in the paper)
# λ(1) = -0.5, λ(2) = 0.5, λ(3) = 1.0, λ(4) = -1.0
whittle_index = np.array([-0.5, 0.5, 1.0, -1.0])

# =============================================================================
# 2. Simulation functions
# =============================================================================
def step(states, active_arms):
    """
    Advance all arms one time step.
    states      : array of current states (integers 0..3)
    active_arms : boolean array of length I, True = active (use P1), False = passive (P0)
    returns     : new states and total reward earned in this step
    """
    new_states = np.empty(I, dtype=int)
    total_reward = 0.0
    for i in range(I):
        s = states[i]
        if active_arms[i]:
            trans = P1[s]
        else:
            trans = P0[s]
        # sample next state
        new_s = np.random.choice(num_states, p=trans)
        new_states[i] = new_s
        total_reward += R[new_s]   # reward is earned when entering new state? 
    # In the problem description, rewards are associated with state-action pairs,
    # but since R depends only on state (and not action) we can also define
    # reward as R(state_after). However the example says "rewards R(1,a)=-1, R(4,a)=1 ..."
    # Usually in RMAB, the reward is accrued at the current state before transition,
    # or after? The Whittle index calculation uses the same cost structure. 
    # The paper discusses average reward "per time step" and the analysis of the "priority to state 3"
    # policy uses the state distribution and transitions. I'll assume reward is earned for
    # the state *after* the transition, consistent with the analysis:
    # "a project that has just transitioned to 4 ... will stay an average of 2 time steps at state 4 ...
    # This yields an average reward of 50 per time step." If the reward were on pre‑transition state,
    # the calculation would differ. The indices were derived for a certain formulation; 
    # we follow the same logic. For simplicity, I'll take reward as R(new_state).
    return new_states, total_reward

def whittle_policy(states):
    """
    Returns a boolean array indicating which arms are active (True = active).
    Activates exactly K arms with the highest Whittle index (break ties at random).
    """
    indices = whittle_index[states]
    # Sort arms by index descending
    # If there are ties, we need a deterministic but fair tie‑breaker.
    # Here we shuffle indices to get random tie‑breaking.
    order = np.lexsort((np.random.random(I), -indices))  # stable sort by -index then random
    active = np.zeros(I, dtype=bool)
    active[order[:K]] = True
    return active

def state3_policy(states):
    """
    Activates K arms that are in state 3 (index 2, original state 3).
    If fewer than K are in state 3, fill the rest with next best state
    according to Whittle priority (state 2, then 1, then 4).
    """
    priority = [2, 1, 0, 3]   # best to worst
    active = np.zeros(I, dtype=bool)
    remaining = K
    for p in priority:
        candidates = np.where(states == p)[0]
        n = min(remaining, len(candidates))
        active[candidates[:n]] = True
        remaining -= n
        if remaining == 0:
            break
    return active

def simulate(policy_func, T, warmup=2000, initial_states=None):
    """
    Run simulation for T steps after warmup.
    Returns average total reward per step and average reward per arm per step.
    """
    if initial_states is None:
        states = np.random.randint(0, num_states, size=I)
    else:
        states = initial_states.copy()
    # Warmup
    for _ in range(warmup):
        active = policy_func(states)
        states, _ = step(states, active)
    # Data collection
    total_reward = 0.0
    for _ in range(T):
        active = policy_func(states)
        states, rew = step(states, active)
        total_reward += rew
    avg_total = total_reward / T
    avg_per_arm = avg_total / I
    return avg_total, avg_per_arm

# =============================================================================
# 3. Run simulations
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    T_sim = 50000
    warmup = 5000

    print("=== Whittle index policy ===")
    avg_total_w, avg_arm_w = simulate(whittle_policy, T_sim, warmup)
    print(f"Average total reward per step: {avg_total_w:.3f}")
    print(f"Average reward per arm per step : {avg_arm_w:.3f}")

    print("\n=== State‑3 priority policy (heuristic) ===")
    avg_total_s3, avg_arm_s3 = simulate(state3_policy, T_sim, warmup)
    print(f"Average total reward per step: {avg_total_s3:.3f}")
    print(f"Average reward per arm per step : {avg_arm_s3:.3f}")

    print("\n(Expected per‑arm reward ≈ 0.1 according to the paper)")

    # Optional: check that Whittle indices indeed represent the priority order
    print("\nWhittle indices (state: index):")
    for s in range(num_states):
        print(f"  state {s+1} (orig {s+1}): λ = {whittle_index[s]:.1f}")