"""
Basic test to compare with precomputed values.
"""
import src.markovianbandit.markovianbandit as bandit
import numpy as np
import time


def compute_Gittins_sonin_algorithm(P, R, discount_factor):
    """
    return Gittins indices using State Elimination method introduced by Isaac Sonin

    This function implements the algorithm described in section 24.3.2 of
    http://www.ece.mcgill.ca/~amahaj1/projects/bandits/book/2013-bandit-computations.pdf
    Equation numbers refer to this paper.
    """
    nb_states = R.shape[0]
    states = list(range(0, nb_states))
    Gidx = np.zeros(nb_states, dtype=np.double)
    # we first find the maximum index
    alpha = np.argmax(R)
    Gidx[alpha] = R[alpha]
    Continue_set = [alpha]
    Q = np.copy(P)
    d = np.copy(R)
    b = (1 - discount_factor) * np.ones(nb_states, dtype=np.double)
    # below is "recursion step"
    for k in range(1, nb_states):
        lamb = discount_factor / (1 - discount_factor * Q[alpha, alpha])
        Stop_set = list(np.setdiff1d(states, Continue_set))
        for x in Stop_set:
            # below is Equation (6)
            coef = lamb * Q[x, alpha]
            d[x] += coef * d[alpha]
            b[x] += coef * b[alpha]
            # below is Equation (5)
            for y in Stop_set:
                Q[x, y] += coef * Q[alpha, y]
        d[alpha] = -np.inf
        # we now compute the argmax and the corresponding index
        alpha = np.argmax(d / b)
        Gidx[alpha] = (1 - discount_factor) * d[alpha] / b[alpha]
        Continue_set.append(alpha)
    return Gidx



def generate_data():
    dim = 10
    for i in range(10):
        np.savetxt('tests/outputs/save_values_{}.npy'.format(i),
                    compute_for_random_example(dim+i, return_indices=True, seed=i))

def test():
    dim = 10
    for i in range(10):
        computed_indices = np.loadtxt('tests/outputs/save_values_{}.npy'.format(i))
        for number_of_updates in [0, 2, 5]:
            new_indices = compute_for_random_example(dim+i, seed=i)
            assert np.allclose(computed_indices, new_indices)

def test_gittins():
    dim = 10
    model = bandit.random_rested(dim)
    P0,P1,R0,R1 = model.get_P0P1R0R1()
    for discount in [.5, .6, .7]:
        our_implem = model.gittins_indices(discount=discount)
        sonin_algo = compute_Gittins_sonin_algorithm(P1, R1, discount)
        assert np.allclose(our_implem, sonin_algo)

def test_multichain():
    P0 = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
        ])
    P1 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
        ])
    R0 = np.array([1.0, 1.0, 1.0, 0.0])
    R1 = np.array([0.0, 0.0, 0.0, 0.0])
    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    indices = model.whittle_indices(check_indexability=True, discount=1, number_of_updates=0)
    print(indices)

def test_multichain_indexable_detected():
    P0 = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
        ])
    P1 = np.array([
        [0.0, 1.0],
        [0.0, 1.0]
        ])
    R0 = np.array([0.0, 1.0])
    R1 = np.array([1.0, 1.0])
    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    indices = model.whittle_indices(check_indexability=True, discount=1, number_of_updates=0)
    print(indices)

def test_multichain_indexable_not_detected():
    P1 = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
        ])
    P0 = np.array([
        [0.0, 1.0],
        [0.0, 1.0]
        ])
    R0 = np.array([0.0, 0.0])
    R1 = np.array([0.0, 1.0])
    model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
    indices = model.whittle_indices(check_indexability=True, discount=1, number_of_updates=0)
    print(indices)

def compute_for_random_example(dim, check_indexability=True, number_of_updates=0, return_time=False, seed=None):
    model = bandit.random_restless(dim, seed=seed)
    tf = time.time()
    indices = model.whittle_indices(check_indexability=check_indexability, discount=1, number_of_updates=number_of_updates)
    if return_time:
        return time.time()-tf
    return indices


#test()
#test_gittins()
#test_multichain()
#test_multichain_indexable_detected()
#test_multichain_indexable_not_detected()
#generate_data()
