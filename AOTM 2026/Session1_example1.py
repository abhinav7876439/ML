# from pyomo.environ import ConcreteModel, Var, Binary, Objective, Constraint
# from pyomo.environ import maximize, SolverFactory, value
from pyomo.environ import *

# Create model
model = ConcreteModel()

# Two binary variables
model.x1 = Var(domain=Binary)
model.x2 = Var(domain=Binary)

# Objective: maximize total benefit
model.obj = Objective(
    expr= 10 * model.x1 + 6 * model.x2,
    sense=maximize
)

# Constraint: select at most one project
model.constraint = Constraint(
    expr=3*model.x1 + 4*model.x2 <= 1
)

# Create SCIP solver
solver = SolverFactory("scip")

# Check SCIP availability
print("SCIP available:", solver.available())

# Solve
results = solver.solve(model, tee=True)

# Display results
print("\nSolution:")
print("x1 =", value(model.x1))
print("x2 =", value(model.x2))
print("Objective =", value(model.obj))