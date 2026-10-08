#!pip install gurobipy

import gurobipy as gp
from gurobipy import GRB

m = gp.Model()
x = m.addVar(lb=0, name="x")
y = m.addVar(lb=0, vtype=GRB.INTEGER, name="y")

m.setObjective(x*x + y, GRB.MINIMIZE)
m.addConstr(3*x + 2*y >= 8)
m.addConstr(x + 2*y >= 6)

m.optimize()

print("x =", x.X)
print("y =", y.X)
print("obj =", m.ObjVal)