from pyomo.environ import ConcreteModel, ConstraintList, Var, Integers, Objective, Constraint, Reals, NonNegativeIntegers
from pyomo.environ import minimize, maximize, SolverFactory, value
import pandas as pd
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the CSV file
csv_file_path = os.path.join(script_dir, 'transportation_100x60_kroncker_block_diagonal.csv')

# Read the CSV using the absolute path
data = pd.read_csv(csv_file_path, header=None)

arr = data.to_numpy()


# ---------------------------------------------------------------
arr = arr[1:, 1:]

print(arr)
print("\n--------------------\n")

# Cost matrix: all rows except the last (Demand row),
#             all columns except the last (Supply column)
arr2 = arr[:-1, :-1].astype(float)
print(arr2)

print("\nCost matrix shape:", arr2.shape)   # should be (100, 60)
print("Number of supplies:", arr[:-1, -1].shape[0])
print("Number of demands :", arr[-1, :].shape[0])
print("Supply sum:", arr[:-1, -1].astype(float).sum())
print("Demand sum:", arr[-1, :].astype(float).sum())   # should equal Supply sum

# Create model
model = ConcreteModel()

model.arr1 = Var(
    range(100),
    range(60),
    domain=NonNegativeIntegers
)

# Objective: minimize total cost
model.obj = Objective(
    expr=sum(
        arr2[i, j] * model.arr1[i, j]
        for i in range(100)
        for j in range(60)
    ),
    sense=minimize
)

model.constraints = ConstraintList()

# Supply constraints (each row sums to its supply)
for i in range(100):
    model.constraints.add(
        sum(model.arr1[i, j] for j in range(60)) == float(arr[i, -1])
    )

# Demand constraints (each column sums to its demand)
for j in range(60):
    model.constraints.add(
        sum(model.arr1[i, j] for i in range(100)) == float(arr[-1, j])
    )

print("\nNumber of variables:", len(list(model.component_data_objects(Var))))
print("Number of constraints:", len(list(model.component_data_objects(Constraint))))

# Create SCIP solver
solver = SolverFactory("gurobi")  # gurobi

# Check SCIP availability
print("SCIP available:", solver.available())

# Solve
results = solver.solve(model, tee=True)

print("Solver status:", results.solver.status)
print("Termination condition:", results.solver.termination_condition)

# Display results
print("\nSolution:")
print("Objective =", value(model.obj))

