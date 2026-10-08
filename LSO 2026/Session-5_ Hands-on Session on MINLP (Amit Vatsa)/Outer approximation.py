#!pip install gurobipy

import gurobipy as gp
from gurobipy import GRB

# -----------------------------
# Problem:
# min x^2 + y
# s.t. 3x + 2y >= 8
#      x  + 2y >= 6
#      x >= 0, y >= 0, y integer
# -----------------------------

tol = 1e-6

# =========================================================
# METHOD 1: Inefficient OA-KCP
# Adds tangent cut at every master solution
# =========================================================
print("===== INEFFICIENT OA-KCP =====\n")

m1 = gp.Model()
m1.Params.OutputFlag = 0

x1 = m1.addVar(lb=0, ub=6, name="x")
y1 = m1.addVar(vtype=GRB.INTEGER, lb=0, name="y")
theta1 = m1.addVar(lb=0, name="theta")

m1.setObjective(theta1 + y1, GRB.MINIMIZE)
m1.addConstr(3*x1 + 2*y1 >= 8)
m1.addConstr(x1 + 2*y1 >= 6)

LB = -1e20
UB = 1e20
it = 1

while True:
    m1.optimize()

    xk = x1.X
    yk = round(y1.X)
    thetak = theta1.X
    LB = m1.ObjVal
    UB = min(UB, xk*xk + yk)

    print(f"Iteration {it}: LB = {LB:.6f}, x = {xk:.6f}, y = {yk}, theta = {thetak:.6f}, UB = {UB:.6f}")

    if abs(UB - LB) <= tol:
        print("STOP as LB = UB\n")
        break

    # tangent cut at current xk
    # theta >= xk^2 + 2*xk*(x - xk) = 2*xk*x - xk^2
    m1.addConstr(theta1 >= 2*xk*x1 - xk*xk)
    print(f"Cut added: theta >= {2*xk:.6f} x {(-xk*xk):+.6f}\n")

    it += 1


# =========================================================
# METHOD 2: True OA
# Solve master -> fix y -> solve NLP -> add one cut
# =========================================================
print("===== TRUE OA =====\n")

m2 = gp.Model()
m2.Params.OutputFlag = 0

x2 = m2.addVar(lb=0, ub=6,name="x")
y2 = m2.addVar(vtype=GRB.INTEGER, lb=0, name="y")
theta2 = m2.addVar(lb=0, name="theta")

m2.setObjective(theta2 + y2, GRB.MINIMIZE)
m2.addConstr(3*x2 + 2*y2 >= 8)
m2.addConstr(x2 + 2*y2 >= 6)

LB = -1e20
UB = 1e20
it = 1

while True:
    m2.optimize()

    xk = x2.X
    yk = round(y2.X)
    thetak = theta2.X
    LB = m2.ObjVal

    print(f"Iteration {it}: Solve master -> LB = {LB:.6f}, x = {xk:.6f}, y = {yk}, theta = {thetak:.6f}")

    # NLP with y fixed
    sub = gp.Model()
    sub.Params.OutputFlag = 0
    xs = sub.addVar(lb=0, ub=6,name="x")
    sub.setObjective(xs*xs + yk, GRB.MINIMIZE)
    sub.addConstr(3*xs + 2*yk >= 8)
    sub.addConstr(xs + 2*yk >= 6)
    sub.optimize()

    xbar = xs.X
    UB = min(UB, sub.ObjVal)

    print(f"             Solve NLP with y = {yk} -> x = {xbar:.6f}, UB = {UB:.6f}")

    if abs(UB - LB) <= tol:
        print("STOP as LB = UB")
        break

    # add tangent cut at NLP optimum xbar
    m2.addConstr(theta2 >= 2*xbar*x2 - xbar*xbar)
    print(f"Cut added: theta >= {2*xbar:.6f} x {(-xbar*xbar):+.6f}\n")

    it += 1