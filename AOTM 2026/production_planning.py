from pyomo.environ import ConcreteModel,ConstraintList, Var, Binary, Integers, Objective, Constraint
from pyomo.environ import maximize, SolverFactory, value


# Create model
model = ConcreteModel()

# Two binary variables
model.x1 = Var(domain=Integers) #model.x1 = Var(within=Integers, bounds=(0,1)) 
model.x2 = Var(domain=Integers)

# Objective: maximize total benefit
model.obj = Objective(
    expr=2 * model.x1 + 3 * model.x2,
    sense=maximize
)

# Constraint: select at most one project
# model.constraint = Constraint(
#     expr=model.x1 + model.x2 <= 1
# )

model.Constraints= ConstraintList()

model.Constraints.add(model.x1<= 4000)
model.Constraints.add(model.x2<= 3500)
model.Constraints.add(model.x1+model.x2<= 5000)
model.Constraints.add(model.x1+2*model.x2<= 8000)
model.Constraints.add(model.x1>= 0)
model.Constraints.add(model.x2>= 0)

# Create SCIP solver
solver = SolverFactory("scip") #SolverFactory("gurobi")

# Check SCIP availability
# print("SCIP available:", solver.available())

# Solve
results = solver.solve(model, tee=True)

# Display results
print("\nSolution:")
print("x1 =", value(model.x1))
print("x2 =", value(model.x2))
print("Objective =", value(model.obj))