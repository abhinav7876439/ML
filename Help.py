import numpy as np
import pandas as pd


def build_simple_mdp(N, p, C, K):
    """
    Simple MDP with states {0,...,N}
    Actions: 0 (passive), 1 (active)
    Returns: P0, P1, R0, R1
    """
    num_states = N + 1
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)

    # Active transitions
    for s in range(num_states):
        P1[s, 0] = 1.0
        R1[s] = C

    # Passive transitions
    for s in range(num_states):
        if s < N:
            P0[s, s+1] = p[s]       # survival
            P0[s, 0] = 1 - p[s]     # breakdown
            R0[s] = K * (1 - p[s])
        else:
            P0[s, 0] = 1.0          # sure breakdown
            R0[s] = K

    return P0, P1, R0, R1

def build_extended_mdp(N, p, C, K, C_switch):
    num_states = N + 2
    P0 = np.zeros((num_states, num_states))
    P1 = np.zeros((num_states, num_states))
    R0 = np.zeros(num_states)
    R1 = np.zeros(num_states)

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

    return P0, P1, R0, R1


def print_transition_matrix_scalar(P, action_name):
    print("="*70)
    print(f"Transition matrix for action {action_name} (scalar indices)")
    print("="*70)
    indices = list(range(P.shape[0]))
    df = pd.DataFrame(P, index=indices, columns=indices)
    print(df)


def print_reward_vector_scalar(R, action_name):
    print("="*70)
    print(f"Reward vector for action {action_name} (scalar indices)")
    print("="*70)
    indices = list(range(len(R)))
    df = pd.DataFrame(R, index=indices, columns=[action_name])
    print(df)


def print_state_mapping(N):
    print("="*70)
    print("State index mapping (scalar → extended state)")
    print("="*70)
    print("0 -> (0,1)")
    print("1 -> (0,0)")
    for i in range(1, N+1):
        print(f"{i+1} -> ({i},0)")


# Example usage
N = 4
p = np.sort(np.random.uniform(0, 1.0, N))[::-1]
p_full = np.append(p, 0.0)
C, K, C_switch = 5.0, 500.0, 1.0

print("="*70)
print(f"Survival probabilities p[0..{N}]: {np.round(p_full,3)}")
print("="*70)

P0_ext, P1_ext, R0_ext, R1_ext = build_extended_mdp(N, p_full, C, K, C_switch)

# Print mapping
print_state_mapping(N)

# Print matrices and vectors with scalar indices
print_transition_matrix_scalar(P0_ext, "Passive (a=0)")
print_reward_vector_scalar(R0_ext, "Passive (a=0)")

print_transition_matrix_scalar(P1_ext, "Active (a=1)")
print_reward_vector_scalar(R1_ext, "Active (a=1)")





# import numpy as np

# # Parameters (customize here)
# N = 5
# C = 10.0          # Active base cost
# K = 100.0         # Breakdown cost
# C_switch = 5   # Switch cost (passive -> active)
# ps = [0.99, 0.95, 0.90, 0.80, 0.70]  # p_0 to p_{N-1}
# p = np.array(ps + [0.0])  # p_N = 0
# gamma_disc = 0.95  # Discount for discounted case
# gamma_high = 0.999  # High discount approx for average cost

# num_states_ext = N + 2  # (0,0)=0, (0,1)=1, (1,0)=2, ..., (N,0)=N+1
# num_states_simple = N + 1

# print("Parameters: N={}, C={}, K={}, C_switch={}, p={}".format(N, C, K, C_switch, ps))
# print("num_states_ext =", num_states_ext)

# # =====================================
# # Build P0, P1, R0, R1 for EXTENDED MDP
# # =====================================
# P0_ext = np.zeros((num_states_ext, num_states_ext))
# P1_ext = np.zeros((num_states_ext, num_states_ext))
# R0_ext = np.zeros(num_states_ext)
# R1_ext = np.zeros(num_states_ext)
# state_info = []
# for s in range(num_states_ext):
#     if s == 0:
#         x, prev_a = 0, 0
#     elif s == 1:
#         x, prev_a = 0, 1
#     else:
#         x, prev_a = s - 1, 0
#     state_info.append((x, prev_a))

#     # Active a=1: always to (0,1) id=1, cost depends on prev_a
#     P1_ext[s, 1] = 1.0
#     R1_ext[s] = C if prev_a == 1 else C + C_switch

#     # Passive a=0
#     px = p[x]
#     # Breakdown to (0,0) id=0
#     P0_ext[s, 0] = 1 - px
#     R0_ext[s] += (1 - px) * K
#     # Survival to (min(x+1,N), 0)
#     if px > 0.0:
#         next_x = min(x + 1, N)
#         next_s = next_x + 1  # (1,0)=2, (2,0)=3, ..., (N,0)=N+1
#         P0_ext[s, next_s] = px

# print("\nExtended MDP matrices ready (P0_ext.shape =", P0_ext.shape, ")")
# print("R0_ext =", np.round(R0_ext, 3))
# print("R1_ext =", np.round(R1_ext, 3))
# print("Example P0_ext[0] (from (0,0)): surv p0 to 2, break 1-p0 to 0:", np.round(P0_ext[0], 3))
# print("Example P0_ext[1] (from (0,1)): same:", np.round(P0_ext[1], 3))

# # =====================================
# # Value Iteration function (works for both simple/extended)
# # =====================================
# def value_iteration(P0, P1, R0, R1, gamma, tol=1e-6, max_iter=10000):
#     ns = len(R0)
#     V = np.zeros(ns)
#     policy = np.zeros(ns, dtype=int)
#     for it in range(max_iter):
#         Q0 = R0 + gamma * np.dot(P0, V)
#         Q1 = R1 + gamma * np.dot(P1, V)
#         Vnew = np.minimum(Q0, Q1)
#         policy = np.argmin(np.stack([Q0, Q1]), axis=0)
#         delta = np.max(np.abs(Vnew - V))
#         V = Vnew.copy()
#         if delta < tol:
#             print(f"  Converged in {it+1} iters (delta={delta:.2e})")
#             break
#     return V, policy

# # Extended MDP optimal policy/values
# V_ext, pi_ext = value_iteration(P0_ext, P1_ext, R0_ext, R1_ext, gamma_disc)
# print("\n=== Extended MDP (gamma={:.3f}) ===".format(gamma_disc))
# for s in range(num_states_ext):
#     x, pa = state_info[s]
#     print("State ({},{}): pi={} V={:.4f}".format(x, pa, pi_ext[s], V_ext[s]))

# # =====================================
# # SIMPLE MDP matrices & value iter
# # =====================================
# P0_simple = np.zeros((num_states_simple, num_states_simple))
# P1_simple = np.zeros((num_states_simple, num_states_simple))
# R0_simple = np.zeros(num_states_simple)
# R1_simple = np.full(num_states_simple, C)
# for s in range(num_states_simple):
#     P1_simple[s, 0] = 1.0  # to state 0
#     px = p[s]
#     P0_simple[s, 0] = 1 - px
#     R0_simple[s] += (1 - px) * K
#     next_s = min(s + 1, N)
#     P0_simple[s, next_s] = px

# V_simple, pi_simple = value_iteration(P0_simple, P1_simple, R0_simple, R1_simple, gamma_disc)
# print("\n=== Simple MDP (gamma={:.3f}) ===".format(gamma_disc))
# for s in range(num_states_simple):
#     print("State {}: pi={} V={:.4f}".format(s, pi_simple[s], V_simple[s]))

# # =====================================
# # CLOSED-FORM DISCOUNTED WHITTLE INDICES (for EXTENDED MDP)
# # =====================================
# def compute_whittle_closed(p, K, C, C_switch, beta, N):
#     G = np.zeros(N + 2)
#     cum_prod = 1.0
#     for k in range(N + 1):
#         H_k = beta ** (k + 1) * cum_prod
#         G[k + 1] = G[k] + (1 - p[k]) * H_k
#         cum_prod *= p[k]
#     whittle = np.zeros(N + 2)
#     for k in range(N + 1):
#         num = 1 - G[k + 1] - p[k] * (1 - beta * G[k])
#         den = 1 - G[k + 1] - beta * p[k] * (1 - G[k])
#         ratio = num / den if abs(den) > 1e-12 else 0.0
#         stuff = K * ratio
#         whittle[k] = stuff - C - C_switch  # (k,0)
#     whittle[1] = K * ( (1 - G[1] - p[0]*(1-beta*G[0])) / (1 - G[1] - beta*p[0]*(1-G[0])) ) - C  # Override for (0,1)
#     return whittle

# whittle_disc_closed = compute_whittle_closed(p, K, C, C_switch, gamma_disc, N)
# print("\n=== Discounted Whittle Indices (CLOSED FORM, gamma={:.3f}) ===".format(gamma_disc))
# for s in range(num_states_ext):
#     x, pa = state_info[s]
#     print("({},{}) ID{}: {:.4f}".format(x, pa, s, whittle_disc_closed[s]))

# # =====================================
# # NUMERICAL DISCOUNTED WHITTLE (verification)
# # =====================================
# def whittle_numerical_one_state(s_start, P0, P1, R0, R1, gamma, tol_m=1e-5):
#     m_lo, m_hi = -10000.0, 10000.0
#     for _ in range(80):
#         m_mid = (m_lo + m_hi) / 2
#         R0_sub = R0 + m_mid
#         _, _ = value_iteration(P0, P1, R0_sub, R1, gamma, tol=1e-5)  # Warmup
#         V, _ = value_iteration(P0, P1, R0_sub, R1, gamma, tol=1e-6)
#         Q0_sub = R0_sub + gamma * np.dot(P0, V)
#         Q1 = R1 + gamma * np.dot(P1, V)
#         diff = Q0_sub[s_start] - Q1[s_start]
#         if abs(diff) < tol_m:
#             return m_mid
#         if diff < 0:
#             m_lo = m_mid  # Increase m
#         else:
#             m_hi = m_mid
#     return m_mid

# print("\n=== Discounted Whittle NUMERICAL (select states, gamma={:.3f}) ===".format(gamma_disc))
# for s in range(num_states_ext):  # All, since small
#     w_num = whittle_numerical_one_state(s, P0_ext, P1_ext, R0_ext, R1_ext, gamma_disc)
#     print("({},{}) ID{}: {:.4f} (closed: {:.4f}, diff={:.2e})".format(
#         *state_info[s], s, w_num, whittle_disc_closed[s], w_num - whittle_disc_closed[s]))

# # =====================================
# # APPROX AVERAGE COST WHITTLE (high gamma closed + numerical)
# # =====================================
# print("\n=== Approx Average Cost Whittle (gamma_high={:.3f}) ===".format(gamma_high))
# whittle_avg_closed = compute_whittle_closed(p, K, C, C_switch, gamma_high, N)
# for s in range(num_states_ext):
#     x, pa = state_info[s]
#     print("({},{}) ID{}: {:.4f}".format(x, pa, s, whittle_avg_closed[s]))

# print("\nNumerical approx average (select states):")
# for s in [0, 1, 2, num_states_ext-1]:
#     w_num_avg = whittle_numerical_one_state(s, P0_ext, P1_ext, R0_ext, R1_ext, gamma_high)
#     print("({},{}) ID{}: {:.4f} (closed: {:.4f})".format(*state_info[s], s, w_num_avg, whittle_avg_closed[s]))

# print("\nP0_ext, P1_ext, R0_ext, R1_ext ready for packages/other use.")
# print("Example: save with np.save('P0_ext.npy', P0_ext) etc.")