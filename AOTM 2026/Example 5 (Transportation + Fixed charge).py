from pyomo.environ import ConcreteModel, ConstraintList, Var, Integers, Objective, Constraint, Reals, NonNegativeIntegers
from pyomo.environ import minimize, maximize, SolverFactory, value
import pandas as pd
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the CSV file
csv_file_path = os.path.join(script_dir, 'transportation_50x30_balanced.csv')

# Construct the full path to the CSV file
df = os.path.join(script_dir, 'transportation_50x30_fixed_costs.csv')

# Read the CSV using the absolute path
data = pd.read_csv(csv_file_path, header=None)

df_fixed_costs = pd.read_csv(df, header=None)
arr = data.to_numpy()
arr_fixed_costs = df_fixed_costs.to_numpy()

# Drop header row and the first column (Source labels)
arr = arr[1:, 1:]

# Cost matrix: 50 supply rows x 30 demand columns
cost = arr[:-1, :-1].astype(float)

# Supply values: last column, all supply rows
supply = arr[:-1, -1].astype(float)

# Demand values: last row, all demand columns, excluding the final total column
demand = arr[-1, :-1].astype(float)

print(cost)
print("\n--------------------\n")


print("\nCost matrix shape:", cost.shape)
print("Number of supplies:", len(supply))
print("Number of demands :", len(demand))
print("Supply sum:", supply.sum())
print("Demand sum:", demand.sum())

# Create model
model = ConcreteModel()

model.x = Var(
    range(50),
    range(30),
    domain=NonNegativeIntegers,
)


# Objective: minimize total cost
model.obj = Objective(
    expr=sum(
        cost[i, j] * model.x[i, j]
        for i in range(50)
        for j in range(30)
    ),
    sense=minimize,
)

model.constraints = ConstraintList()

# Supply constraints (each row sums to its supply)
for i in range(50):
    model.constraints.add(
        sum(model.x[i, j] for j in range(30)) == supply[i]
    )

# Demand constraints (each column sums to its demand)
for j in range(30):
    model.constraints.add(
        sum(model.x[i, j] for i in range(50)) == demand[j]  # # Balanced transportation problem
    )

print("\nNumber of variables:", len(list(model.component_data_objects(Var))))
print("Number of constraints:", len(list(model.component_data_objects(Constraint))))

# Solve
solver = SolverFactory("gurobi")   # or "scip", "cbc", "glpk"
print("Solver:", solver.name, "| available:", solver.available())

results = solver.solve(model, tee=True)

print("Solver status:", results.solver.status)
print("Termination condition:", results.solver.termination_condition)

print("\nSolution:")
print("Objective =", value(model.obj))

## Optional: print non-zero shipments
# print("\nNon-zero shipments:")
# for i in range(50):
#     for j in range(30):
#         val = value(model.x[i, j])
#         if val > 0:
#             print(f"x[{i+1},{j+1}] = {val}")