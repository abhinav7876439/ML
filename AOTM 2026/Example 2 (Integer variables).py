"""
Production Planning with Integer Variables — Pyomo + (SCIP, Gurobi)
==========================================================
Maximize profit for two products subject to:
  - individual demand caps
  - two shared resource limits
Decision variables are INTEGER (whole units produced).
"""

from pyomo.environ import (
    ConcreteModel, ConstraintList, Var, Objective, Constraint,
    NonNegativeReals, Integers, maximize, ConstraintList,
    SolverFactory, value,
)
from pyomo.opt import SolverStatus, TerminationCondition


# ---------------------------------------------------------------
# Create model
# ---------------------------------------------------------------
model = ConcreteModel()


# ---------------------------------------------------------------
# Decision variables (INTEGER — whole units produced)
#    x[1] = units of product 1, x[2] = units of product 2
# ---------------------------------------------------------------
model.x = Var([1, 2], within=Integers, bounds=(0, None))

# For the continuous (LP) relaxation instead, use:
# model.x = Var([1, 2], within=NonNegativeReals)

# or
# model.x1 = Var(within=NonNegativeReals)
# model.x2 = Var(within=NonNegativeReals)


# ---------------------------------------------------------------
# Objective: maximize total profit contribution
#    Product 1 -> $2/unit, Product 2 -> $3/unit
# ---------------------------------------------------------------
model.obj = Objective(
    expr=2 * model.x[1] + 3 * model.x[2],
    sense=maximize,
)


# ---------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------
# Individual demand caps
# model.c1 = Constraint(expr=model.x[1] <= 4000)
# model.c2 = Constraint(expr=model.x[2] <= 3500)

# # Shared resource A (e.g., machine hours)
# model.c3 = Constraint(expr=model.x[1] + model.x[2] <= 5000)

# # Shared resource B (product 2 uses twice as much per unit)
# model.c4 = Constraint(expr=model.x[1] + 2 * model.x[2] <= 8000)



# ---------------------------------------------------------------
# Constraints via ConstraintList
# ---------------------------------------------------------------
model.cons = ConstraintList()  # import ConstraintList from pyomo.environ

# Each add() appends a new constraint.
# The list auto names them cons[1], cons[2], cons[3], cons[4], ...
model.cons.add(model.x[1] <= 4000)                    # cons[1]  demand P1
model.cons.add(model.x[2] <= 3500)                    # cons[2]  demand P2
model.cons.add(model.x[1] + model.x[2] <= 5000)       # cons[3]  resource A
model.cons.add(model.x[1] + 2 * model.x[2] <= 8000)   # cons[4]  resource B


# ---------------------------------------------------------------
# Solve
# ---------------------------------------------------------------
solver = SolverFactory("gurobi")     # or "scip", "cbc", "glpk"
print("Solver:", solver.name, "| available:", solver.available())

results = solver.solve(model, tee=True)



# ---------------------------------------------------------------
# 6. Report results
# ---------------------------------------------------------------
print("\n=== Solver Status ===")
print("Status:       ", results.solver.status)
print("Termination:  ", results.solver.termination_condition)

if (results.solver.status == SolverStatus.ok and
        results.solver.termination_condition == TerminationCondition.optimal):

    print("\n=== Solution (Optimal) ===")
    print(f"x1 (Product 1 units) = {value(model.x[1]):.0f}")
    print(f"x2 (Product 2 units) = {value(model.x[2]):.0f}")
    print(f"Objective (profit)   = {value(model.obj):.2f}")
