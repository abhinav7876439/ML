

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import bisect
import networkx as nx
from scipy.optimize import brentq
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





# ----------------------------------------------------------------------
# Closed‑form discounted indices
# ----------------------------------------------------------------------
def compute_G_H(p, beta):
    N = len(p)
    H = np.zeros(N+1)
    G = np.zeros(N+2)
    prod = 1.0
    for k in range(N+1):
        if k < N:
            H[k] = beta**(k+1) * prod
            prod *= p[k]
        else:
            H[k] = beta**(k+1) * prod
    G[0] = 0.0
    for k in range(N+1):
        p_k = p[k] if k < N else 0.0
        G[k+1] = G[k] + (1 - p_k) * H[k]
    return G, H

def closed_form_indices(p, beta, K, C, C_switch):
    N = len(p)
    G, H = compute_G_H(p, beta)
    p0 = p[0] if N > 0 else 0.0
    w00 = (1 - p0) * K - C - C_switch * (1 - beta)

    if N >= 1:
        G1 = G[1]
        H1 = H[1]
        num = (1 - beta) * (K * G1 + C_switch * H1)
        den = beta * (1 - G1) - H1
        w_active = (num / den) - C if abs(den) > 1e-15 else np.inf
    else:
        w_active = np.inf

    w_passive = np.zeros(N+1)
    w_passive[0] = w00
    for k in range(1, N+1):
        pk = p[k] if k < N else 0.0
        Gk = G[k]
        Hk = H[k]
        denom = (1 - pk * beta) * (1 - Gk) - (1 - pk) * Hk
        if abs(denom) > 1e-15:
            w_passive[k] = K - C - C_switch - (K * pk * (1 - beta)) / denom
        else:
            w_passive[k] = np.inf
    return w_active, w_passive  

# ============================================================================
# Example usage and comparison with the markovianbandit package
# ============================================================================
if __name__ == "__main__":
    # Parameters
    N = 4
    np.random.seed(42)
    p = np.sort(np.random.uniform(0, 1.0, N))[::-1]   # decreasing survival probabilities
    p_full = np.append(p, 0.0)                       # terminal zero for the extended MDP
    C, K = 5.0, 500.0
    C_switch = 1.0
    beta = 0.99
    
    print("=" * 70)
    print(f"Survival probabilities p[0..{N}]: {np.round(p_full, 3)}")
    print("=" * 70)

    P0_ext, P1_ext, R0_ext, R1_ext, state_labels_ext = build_extended_mdp(N, p_full, C, K, C_switch)
    p_raw = p_full[:-1]  # exclude the terminal zero for closed-form indices

    # Print matrices with state labels both vertically and horizontally
    print_transition_matrix(P0_ext, state_labels_ext, "Passive (a=0)")
    print_reward_vector(R0_ext, state_labels_ext, "Passive (a=0)")

    print_transition_matrix(P1_ext, state_labels_ext, "Active (a=1)")
    print_reward_vector(R1_ext, state_labels_ext, "Active (a=1)")

    # Closed-form indices
    print("\n--- Closed Form (Discounted) ---")
    # W_passive, W_active = closed_form_indices(p_full, beta, K, C, C_switch)
    W_active, W_passive = closed_form_indices(p_raw, beta, K, C, C_switch)
    print(f"  Active state (0,1): W = {W_active:.6f}")
    for k in range(N+1):
        print(f"  Passive state ({k},0): W = {W_passive[k]:.6f}")
    




    # Closed‑form indices
    # CORRECT UNPACKING: first returned is active index, second is passive array
    W_active, W_passive = closed_form_indices(p_full, beta, K, C, C_switch)
    
    # Robust extraction of scalar from possible numpy array
    if isinstance(W_active, np.ndarray):
        W_active_scalar = W_active.flat[0]   # get first element
    else:
        W_active_scalar = float(W_active)
    
    print("\nClosed‑form Whittle indices (discounted, beta = {}):".format(beta))
    print(f"  Active state (0,1): W = {W_active_scalar:.6f}")
    for k in range(N+1):
        print(f"  Passive state ({k},0): W = {W_passive[k]:.6f}")

    # Compare with markovianbandit package if available
    if PKG_AVAILABLE:
        # Build the extended MDP using the full p (including terminal zero)
        P0, P1, R0, R1, _ = build_extended_mdp(N, p_full, C, K, C_switch)
        
        # Create restless bandit model
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
        
        # Discounted indices from the package
        pkg_disc = model.whittle_indices(discount=beta)
        pkg_passive = pkg_disc[1:N+2]          # states (0,0) ... (N-1,0)
        pkg_active_special = pkg_disc[0]       # state (0,1)
        
        print("\nPackage discounted indices (beta = {}):".format(beta))
        print(f"  Active state (0,1): W = {pkg_active_special:.6f}")
        for k in range(N+1):
            print(f"  Passive state ({k},0): W = {pkg_passive[k]:.6f}")
        
        # Differences
        print("\nDifferences (closed‑form - package):")
        print(f"  Active (0,1): {W_active_scalar - float(pkg_active_special):.2e}")
        for k in range(N+1):
            diff = W_passive[k] - pkg_passive[k]
            print(f"  Passive ({k},0): {diff:.2e}")
        
        # (Optional) Average‑cost indices (discount=0)
        pkg_avg = model.whittle_indices(discount=0.0)
        pkg_avg_passive = pkg_avg[1:N+2]
        pkg_avg_active_special = pkg_avg[0]
        
        print("\nPackage average‑cost indices:")
        print(f"  Active state (0,1): W = {pkg_avg_active_special:.4f}")
        for k, val in enumerate(pkg_avg_passive):
            print(f"  Passive state ({k},0): W = {val:.4f}")
        print("=" * 60)

