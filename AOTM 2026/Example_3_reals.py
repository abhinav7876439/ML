from pyomo.environ import *


model = ConcreteModel()

# Two binary variables
model.x1 = Var(within=Reals,bounds=(0,float('inf'))) #model.x1 = Var(within=Integers, bounds=(0,1)) 
model.x2 = Var(within=Reals,bounds=(0,float('inf')))

# Objective: maximize total benefit
model.obj = Objective(
    expr=4 * model.x1 + 5 * model.x2,
    sense=minimize
)

# Constraint: select at most one project
# model.constraint = Constraint(
#     expr=model.x1 + model.x2 <= 1
# )

model.Constraints= ConstraintList()

# model.Constraints.add(model.x1<= 4000)
# model.Constraints.add(model.x2<= 3500)
model.Constraints.add(model.x1+4*model.x2>= 5)
model.Constraints.add(3*model.x1+2*model.x2>= 7)
model.Constraints.add(model.x2<= 0)
model.Constraints.add(model.x1>= 2)
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