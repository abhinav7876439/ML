from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals, Integers
from pyomo.environ import maximize, SolverFactory, value

# Create model
model = ConcreteModel()

# Two continuous, nonnegative variables
model.x1 = Var(within=NonNegativeReals)
model.x2 = Var(within=NonNegativeReals)

# model.x1 = Var(within=Integers)
# model.x2 = Var(within=Integers)

# Objective: maximize total benefit
model.obj = Objective(
    expr=2 * model.x1 + 3 * model.x2,
    sense=maximize
)

# Linear constraints
model.c1 = Constraint(expr=model.x1 <= 4000)
model.c2 = Constraint(expr=model.x2 <= 3500)
model.c3 = Constraint(expr=model.x1 + model.x2 <= 5000)
model.c4 = Constraint(expr=model.x1 + 2 * model.x2 <= 8000)



# Create Gurobi solver
solver = SolverFactory("gurobi")  

# Check Gurobi availability
print("Gurobi available:", solver.available())

# Solve
results = solver.solve(model, tee=True)

# Display results
print("\nSolution:")
print("x1 =", value(model.x1))
print("x2 =", value(model.x2))
print("Objective =", value(model.obj))