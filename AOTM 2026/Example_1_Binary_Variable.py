from pyomo.environ import ConcreteModel, Var, Binary, Integers, Objective, Constraint
from pyomo.environ import maximize, SolverFactory, value

# Create model
model = ConcreteModel()

# Two binary variables
# model.x1 = Var(domain=Binary)
# model.x2 = Var(domain=Binary)


model.x1 = Var(within=Integers, bounds=(0,1))
model.x2 = Var(within=Integers, bounds=(0,1))

# Objective: maximize total benefit
model.obj = Objective(
    expr=10 * model.x1 + 6 * model.x2,
    sense=maximize
)

# Constraint: select at most one project
model.constraint = Constraint(
    expr=model.x1 + model.x2 <= 1
)

# Create Gurobi solver
solver = SolverFactory("scip")  # You can also use "scip" if you have it installed

# Check Gurobi availability
print("Gurobi available:", solver.available())

# Solve
results = solver.solve(model, tee=True)

# Display results
print("\nSolution:")
print("x1 =", value(model.x1))
print("x2 =", value(model.x2))
print("Objective =", value(model.obj))