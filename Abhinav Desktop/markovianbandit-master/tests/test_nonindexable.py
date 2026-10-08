"""
Test to see if the code can detect a non-indexable example
"""
import src.markovianbandit.markovianbandit as bandit

def test_nonindexable():
    for seed in range(2751, 2800):
        model = bandit.random_restless(4, seed=seed)
        assert bool(model.is_indexable()) == (False if seed == 2791 else True)
#test_nonindexable()
