from pyomo.environ import *
import pandas as pd
import numpy as np
import re

costs = [
    [90, 80, 75, 70],
    [35, 85, 55, 65],
    [125, 95, 90, 95],
    [45, 110, 95, 115],
  #  [50, 100, 90, 100],
]
num_workers = len(costs)
num_tasks = len(costs[0])

print('Number of workers', num_workers)

print('Number of tasks', num_tasks)

