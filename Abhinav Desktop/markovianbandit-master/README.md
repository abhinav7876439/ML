# Markovian Bandits

This repository contains a python library to compute whittle indices or test indexability of finite-state Markovian bandit problems.

## Installation

To install, just run:
```
pip install markovianbandit-pkg
```

## Example

To test with a few randomly generated examples:
```
import markovianbandit as bandit
model = bandit.random_restless(dim=4, seed=42)
print(model.whittle_indices()) # should print array([ 0.87536099, -0.08765819, -0.15279431, -0.51905682])
print(model.get_P0P1R0R1())
model = bandit.random_restless(4, seed=2791)
print(model.is_indexable()) # should print False

model = bandit.random_rested(dim=4)
print(model.gittins_indices(discount=.8)) # computes gittins index
```

To compute the Whittle or Gittins index from P0 and P1:
```
P0 = [[.5, .5], [.25, .75]]
P1 = [[1, 0], [0.5, 0.5]]
R0 = [0.5, 0.5]
R1 = [2, 1]
model = bandit.restless_bandit_from_P0P1_R0R1(P0, P1, R0, R1)
print(model.whittle_indices())

model = bandit.rested_bandit_from_P1_R1(P1, R1)
print(model.gittins_indices(discount=0.5))
```

Note that the algorithm complexity to compute Whittle index is $O(n^3)$. It can compute indices of bandits in less than one seconds to bandits of up to $1000$ states (i.e. $1000\times 1000$ matrices). The loop below should take less than one second.
```
import time
for dim in [10, 100, 1000]:
    ts = time.time()
    model = bandit.random_restless(dim=dim, seed=42)
    model.whittle_indices()
    print("Model of size", dim, "computed in", time.time() - ts, "seconds")
```

## Authors

Nicolas Gast and Kimang Khun. 

## History version

- 0.4 (July 10, 2024):
  - fixed implementation in find_mu_min() (fix a bug triggered when some indices where equal)

- 0.3 (July 10, 2023): 
  - handle multichain case by testing when matrix inversion fails
  - correct treatment of possibly infinite indices
  - correction of a bug when dividing by 0.
  - addition of an absolute tolerance to be more robust

- 0.2 (Nov 21, 2022)
  - simplification of how the discount is treated
  - function to avoid recomputation of indices
  - new indexability test

- 0.1 (Jan 11, 2021)
  - Many small optimization (code about 50% faster)

- 0.0.5 (Nov 19, 2021). First public release.
## Reference

"Testing Indexability and Computing Whittle and Gittins Index in Subcubic Time" by Nicolas Gast, Kimang Khun and Bruno Gaujal.
https://arxiv.org/abs/2203.05207 
