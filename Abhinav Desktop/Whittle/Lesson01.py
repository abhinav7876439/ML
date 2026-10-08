import numpy as np
num_states = 5
probas_r = np.zeros((num_states, 2))
print(probas_r)


def uniform_arm_distribution(N, s, activation_fraction=0.2):
    # Step 1: Uniform distribution of arms across states
    arms_per_state = [N // s] * s  # Start with floor division
    remainder = N % s              # Distribute the remainder evenly
    for i in range(remainder):
        arms_per_state[i] += 1

    # Step 2: Uniform activation of 20% of arms
    activation_fraction = activation_fraction  # Percentage to activate
    active_arms = [int(arm * activation_fraction) for arm in arms_per_state]
    fractional_parts = [arm * activation_fraction - int(arm * activation_fraction) for arm in arms_per_state]

    # Handle rounding discrepancy using randomized rounding
    deficit = int(round(sum(arm * activation_fraction for arm in arms_per_state))) - sum(active_arms)
    if deficit > 0:
        indices = np.argsort(-np.array(fractional_parts))[:deficit]
        for idx in indices:
            active_arms[idx] += 1

    return arms_per_state, active_arms




def pre_rounding(states):
    rounded_states = np.floor(states).astype(int)  # Start with the floor of each value
    fractional_parts = states - rounded_states
    deficit = int(round(sum(states))) - sum(rounded_states)  # Adjust to maintain the total

    # Randomly round up based on fractional parts
    if deficit > 0:
        indices = np.argsort(-fractional_parts)[:deficit]
        rounded_states[indices] += 1

    return rounded_states



"""Randomized Rounding Code"""

def find_ind(thres, s):
    N = len(s)
    if thres > s[N - 1] - 1e-8:
        return N - 1
    left = 0
    right = N - 1
    current = (left + right) // 2
    while right - left > 1:
        if s[current] > thres:
            right = current
        else:
            left = current
        current = (left + right) // 2
    return current


def placement(c, y):
    N = len(y)
    s = np.zeros(N)
    t = np.zeros(N)
    tau = np.zeros(N)
    Tau = np.ones(N + 1)
    sums = 0.0
    for i in range(N):
        s[i] = sums
        t[i] = sums + y[i]
        tau[i] = t[i] - int(t[i]) if abs(t[i] - round(t[i])) >= 1e-8 else 0
        sums = t[i]
    tau_sort = sorted(set(tau))
    K = len(tau_sort)
    Tau[:K] = tau_sort
    nu = []
    for k in range(K - 1):
        x = np.zeros(N, dtype=int)
        for l in range(c):
            thres = l + Tau[k]
            ind = find_ind(thres, s)
            x[ind] = 1
        nu.append([x, Tau[k + 1] - Tau[k]])
    return nu

def transform(ans1, y):
    card = len(y)
    action = np.zeros(card, dtype=int)
    left = 0
    right = 0
    for i in range(card):
        left = right
        right += y[i][1]
        action[i] = sum(ans1[left:right])
    return list(action)


def rounding(c, y):
    card = len(y)
    z = []
    for i in range(card):
        # Convert y[i][1] to an integer
        z.extend([y[i][0]] * int(round(y[i][1])))
    ans = placement(c, z)
    possible_acts = []
    probas = []
    for item in ans:
        act = transform(item[0], y)
        if act in possible_acts:
            ind = possible_acts.index(act)
            probas[ind] += item[1]
        else:
            possible_acts.append(act)
            probas.append(item[1])
    return possible_acts, probas


def randomized_rounding(probas, states, activation_fraction=0.2):
    N = sum(states)
    target_activation = int(round(activation_fraction * N))  # Total arms to activate
    base = np.zeros(len(probas), dtype=int)
    reduced = []
    c = 0
    for i in range(len(probas)):
        s = int(probas[i][0])
        nb_s = states[s]
        p_s = probas[i][1]
        prod = nb_s * p_s
        base[i] = int(prod)
        reduced.append([s, prod - base[i]])
        c += prod - base[i]
    c = int(round(c))
    acts, prob = rounding(target_activation, reduced)
    possible_acts = []
    for item in acts:
        possible_acts.append(list(base + item))
    return possible_acts, prob