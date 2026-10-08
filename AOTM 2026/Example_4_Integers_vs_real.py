from pyomo.environ import *


model = ConcreteModel()

# Two binary variables
# model.x1 = Var(within=Integers,bounds=(0,3.5)) #model.x1 = Var(within=Integers, bounds=(0,1)) 
# model.x2 = Var(within=Integers,bounds=(0,float('inf')))
model.x1 = Var(within=Reals,bounds=(0,3.5)) #model.x1 = Var(within=Integers, bounds=(0,1)) 
model.x2 = Var(within=Reals,bounds=(0,float('inf')))

# Objective: maximize total benefit
model.obj = Objective(
    expr= model.x1 + 10 * model.x2,
    sense=maximize
)

# Constraint: select at most one project
# model.constraint = Constraint(
#     expr=model.x1 + model.x2 <= 1
# )

model.Constraints= ConstraintList()

model.Constraints.add(model.x1+7*model.x2<= 17.5)
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