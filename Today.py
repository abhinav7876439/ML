import numpy as np
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


def build_mdp_extended(N, p, C, K, c_switch):
    """
    Build extended MDP with failure state based on the corrected reward structure.
    
    States:
        0..N        : (x,0) passive family
        N+1 .. 2N+1 : (x,1) active family (index = N+1 + x)
        2N+2        : failure state F (absorbing, zero cost)
    
    Passive action (a=0):
        - (x,0) for x<N: with prob p[x] -> (x+1,0) (reward 0), with prob 1-p[x] -> (0,0) (reward -K)
        - (N,0): always -> (0,0) (reward -K)
        - (0,1): with prob p[0] -> (1,0) (reward 0), with prob 1-p[0] -> (0,0) (reward -K)
        - (x,1) for x>=1: always -> F (reward 0)
        - F: stays in F (reward 0)
    
    Active action (a=1):
        - (0,1): stays in (0,1) with reward -(C + w) [using C as Cm + w]
        - (x,0) for x>=0: go to (0,1) with reward -(C + c_switch)
        - Note: Active action for (x>=1,1) not defined in original structure
        - F: stays in F (reward 0)
    """
    n_states = 2 * (N + 1) + 1
    F = n_states - 1
    
    P0 = np.zeros((n_states, n_states))
    P1 = np.zeros((n_states, n_states))
    R0 = np.zeros(n_states)
    R1 = np.zeros(n_states)
    
    # ----- Passive action (a=0) -----
    for x in range(N + 1):
        idx_p = x                      # (x,0)
        idx_a = (N + 1) + x            # (x,1)
        
        # (x,0) passive states
        if x < N:
            # (x,0) -> (x+1,0) with prob p[x], reward 0
            # (x,0) -> (0,0) with prob 1-p[x], reward -K
            P0[idx_p, x+1] = p[x]
            P0[idx_p, 0] = 1 - p[x]    # Go to (0,0), NOT F
            R0[idx_p] = -K * (1 - p[x])
        else:  # x == N
            # (N,0) -> (0,0) with prob 1, reward -K
            P0[idx_p, 0] = 1.0
            R0[idx_p] = -K
        
        # (x,1) active states
        if x == 0:
            # (0,1) -> (1,0) with prob p[0], reward 0
            # (0,1) -> (0,0) with prob 1-p[0], reward -K
            P0[idx_a, 1] = p[0]        # Go to (1,0)
            P0[idx_a, 0] = 1 - p[0]    # Go to (0,0)
            R0[idx_a] = -K * (1 - p[0])
        else:  # x >= 1
            # (x>=1,1) -> F with prob 1, reward 0
            P0[idx_a, F] = 1.0
            R0[idx_a] = 0.0
    
    # ----- Active action (a=1) -----
    for x in range(N + 1):
        idx_p = x                      # (x,0)
        idx_a = (N + 1) + x            # (x,1)
        target_active = N + 1          # (0,1)
        
        # From passive states (x,0): switch to active
        # Go to (0,1) with cost C + c_switch
        P1[idx_p, target_active] = 1.0
        R1[idx_p] = -(C + c_switch)
        
        # From active state (0,1): stay active (repair, no switch)
        if x == 0:
            P1[idx_a, target_active] = 1.0
            R1[idx_a] = -C  # C represents Cm + w
        else:
            # From active states (x>=1,1): active action not defined in original structure
            # Set to self-loop with very negative reward to discourage
            P1[idx_a, target_active] = 1.0
            R1[idx_a] = 0.0
    
    # Failure state F: absorbing under both actions, zero reward
    P0[F, F] = 1.0
    R0[F] = 0.0
    P1[F, F] = 1.0
    R1[F] = 0.0
    
    return P0, P1, R0, R1


def build_mdp(N, p, C, K, c_switch=0):
    """
    Build the extended MDP for a single machine with states (x,a_{-1}).
    Returns:
        P0, P1: transition matrices for passive (a=0) and active (a=1)
        R0, R1: immediate cost vectors for passive and active (negative costs)
    """
    n_states = (N+1)*2
    P0 = np.zeros((n_states, n_states))
    P1 = np.zeros((n_states, n_states))
    R0 = np.zeros(n_states)
    R1 = np.zeros(n_states)


    # Active/Passive offset
    # Indices 0 to N: Passive family (0,0) to (N,0)
    # Indices N+1 to 2N+1: Active family (0,1) to (N,1)

    for x in range(N+1):
        idx_p = x               # (x,0), Passive: 0 to N
        idx_a = x + (N+1)       # (x,1), Active: N+1 to 2N+1

        
        # --- Passive Action Dynamics (P0) ---
        # Catastrophic Breakdown (1-px) -> leads to (0,0), index 0
        P0[idx_p, 0] += (1 - p[x])
        P0[idx_a, 0] += (1 - p[x])
        
        # Deterioration (p_x)-> (x+1,0) if possible
        if x < N:
            P0[idx_p, x+1] += p[x]
            P0[idx_a, x+1] += p[x]
        else:
            # At last state, treat deterioration as failure (or stay)
            # Rule: Last state only has catastrophic failure, failure with prob 1 (p[N]=0)
            P0[idx_p, 0] += p[x]
            P0[idx_a, 0] += p[x]


        # Cost for passive: breakdown cost K * (1-p_x)
        R0[idx_p] = -K * (1 - p[x])
        R0[idx_a] = -K * (1 - p[x])

        # Active action (a=1) Dynamics (P1) -> always go to (0,1) i.e. index N+1
        P1[idx_p, N+1] = 1.0
        P1[idx_a, N+1] = 1.0
        R1[idx_p] = -(C + c_switch)   # switching cost for passive family
        R1[idx_a] = -C                 # no switching cost if already active

    return P0, P1, R0, R1


# ------------------------------------------------------------
#  Closed-form expressions
# ------------------------------------------------------------
def compute_G_H(p, beta):
    """ Computes G(k) and H(k) as defined in the paper. """
    N = len(p)
    H = np.zeros(N)
    G = np.zeros(N+1)
    for k in range(N):
        prod = 1.0
        for i in range(k):
            prod *= p[i]
        H[k] = beta**(k+1) * prod
        G[k+1] = G[k] + (1 - p[k]) * H[k]
    return G, H

def closed_form_indices_discounted(p, C, K, beta, c_switch):
    """
    Discounted Whittle indices for states x = 0..N-1 (active states).
    p[0..N-1] are the survival probabilities; p[N] is not used.
    """
    N = len(p)          # p has length N (original N, without terminal zero)
    G, H = compute_G_H(p, beta)
    W = np.zeros(N)
    for k in range(N):
        num = 1 - G[k+1] - p[k] * (1 - beta * G[k])
        den = 1 - G[k+1] - beta * p[k] * (1 - G[k])
        W[k] = K * (num / den) - C - c_switch  # c_switch * (k == 0)
    return W

def compute_T_k_avg(p):
    """
    Expected cycle times T[k] for average cost case.
    T[0] = 1, T[k] = 1 + p[k-1] * T[k-1] for k=1..N.
    """
    N = len(p)
    T = np.zeros(N+1)
    T[0] = 1.0
    for k in range(1, N+1):
        T[k] = 1 + p[k-1] * T[k-1]
    return T

def closed_form_indices_average(p, C, K):
    """
    Average-cost Whittle indices for states x = 0..N-1.
    p[0..N-1] survival probabilities.
    """
    N = len(p)
    T = compute_T_k_avg(p)
    W = np.zeros(N)
    for k in range(N):
        denom = T[k+1] - p[k] * T[k]
        if denom > 0:
            W[k] = K * (1 - p[k] / denom) - C
        else:
            W[k] = K - C   # degenerate case
    return W


# ------------------------------------------------------------
#  Main: run all methods and print results
# ------------------------------------------------------------
def main():
    np.random.seed(42)
    N = 4                     # number of non‑terminal states (0..N-1)
    # Generate decreasing survival probabilities, last is zero (terminal)
    p_full = np.sort(np.random.uniform(0, 1.0, N))[::-1]   # length N
    p_full = np.append(p_full, 0.0)                        # length N+1, p[N]=0
    # For closed‑form we use only the first N entries
    p_closed = p_full[:N]

    print("Survival probabilities p[0..{}]:".format(N-1), np.round(p_closed, 3))
    print("Terminal p[{}] = 0.0".format(N))
    print()

    C = 5.0
    K = 500.0
    c_switch = 0.0
    beta = 0.95

    # 1. Closed‑form indices
    W_disc_closed = closed_form_indices_discounted(p_closed, C, K, beta, c_switch)
    W_avg_closed  = closed_form_indices_average(p_closed, C, K)

    print("=" * 60)
    print("Closed‑form discounted indices (beta = {})".format(beta))
    for x, val in enumerate(W_disc_closed):
        print("  W({},1) = {:.4f}".format(x, val))
        print("  W({},0) = {:.4f}".format(x, val))
    print("\nClosed‑form average‑cost indices")
    for x, val in enumerate(W_avg_closed):
        print("  W({},1) = {:.4f}".format(x, val))
        print("  W({},0) = {:.4f}".format(x, val))
    print("=" * 60)

    # 2. Package comparison (if available)
    if PKG_AVAILABLE:
        # Build MDP with full p (including terminal zero)
        P0, P1, R0, R1 = build_mdp_extended(N, p_full, C, K, c_switch)
        print("P0 shape:", P0.shape)
        print("P0:", P0)
        print("P1 shape:", P1.shape)
        print("P1:", P1)
        print("R0 shape:", R0.shape)
        print("R0:", R0)
        print("R1 shape:", R1.shape)
        print("R1:", R1)
        model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)

        # Discounted indices
        pkg_disc = model.whittle_indices(discount=beta)
        pkg_disc_active = pkg_disc[N+1:2*(N+1)]   # states (x,1) for x=0..N
        # Package returns indices for all states; the terminal active state (N,1) is also computed.
        # For comparison we take only x=0..N-1 (the same as closed‑form)
        pkg_disc_active = pkg_disc_active[:N]

        # Average‑cost indices (discount=0 corresponds to average cost in the package)
        pkg_avg = model.whittle_indices(discount=0.0)
        pkg_avg_active = pkg_avg[N+1:2*(N+1)][:N]

        print("\nPackage discounted indices (beta = {})".format(beta))
        for x, val in enumerate(pkg_disc_active):
            print("  W({},1) = {:.4f}".format(x, val))
            print("  W({},0) = {:.4f}".format(x, val))

        print("\nPackage average‑cost indices")
        for x, val in enumerate(pkg_avg_active):
            print("  W({},1) = {:.4f}".format(x, val))
            print("  W({},0) = {:.4f}".format(x, val))
        print("=" * 60)

        # Comparison for discounted
        print("\nDiscounted indices (closed‑form vs package):")
        print("State | Closed‑form | Package")
        print("-" * 35)
        for x in range(N):
            print(f"  {x:2d}   | {W_disc_closed[x]:10.4f} | {pkg_disc_active[x]:10.4f}")

        # Comparison for average
        print("\nAverage‑cost indices (closed‑form vs package):")
        print("State | Closed‑form | Package")
        print("-" * 35)
        for x in range(N):
            print(f"  {x:2d}   | {W_avg_closed[x]:10.4f} | {pkg_avg_active[x]:10.4f}")

if __name__ == "__main__":
    main()