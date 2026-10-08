!pip install gurobipy

import gurobipy as gp
from gurobipy import GRB

# -----------------------------
# McCormick relaxation
# -----------------------------
m = gp.Model("mccormick")

x = m.addVar(vtype=GRB.INTEGER, lb=2, ub=8, name="x")
y = m.addVar(lb=1, ub=3, name="y")
w = m.addVar(name="w")   # w = x*y

m.setObjective(4*x - 2*y + w, GRB.MINIMIZE)

# xy >= 8 becomes w >= 8
m.addConstr(w >= 8)

# bounds
xL, xU = 2, 8
yL, yU = 1, 3

# McCormick envelope
m.addConstr(w >= xL*y + yL*x - xL*yL)   # w >= 2y + x - 2
m.addConstr(w >= xU*y + yU*x - xU*yU)   # w >= 8y + 3x - 24
m.addConstr(w <= xU*y + yL*x - xU*yL)   # w <= 8y + x - 8
m.addConstr(w <= xL*y + yU*x - xL*yU)   # w <= 2y + 3x - 6

m.optimize()

print("Best solution from McCormick relaxation")
print("x =", x.X)
print("y =", y.X)
print("w =", w.X)
print("objective =", m.ObjVal)

# -----------------------------
# Original nonconvex problem with x = 3
# -----------------------------
m2 = gp.Model("original_nonconvex_x_eq_3")
m2.Params.NonConvex = 2

x2 = m2.addVar(lb=2, ub=8, vtype=GRB.INTEGER, name="x")
y2 = m2.addVar(lb=1, ub=3, name="y")

m2.setObjective(4*x2 - 2*y2 + x2*y2, GRB.MINIMIZE)

m2.addConstr(x2 * y2 >= 8)
m2.addConstr(x2 == 3)

m2.optimize()

print("\nBest solution for original nonconvex problem with x = 3")
print("x =", x2.X)
print("y =", y2.X)
print("objective =", m2.ObjVal)