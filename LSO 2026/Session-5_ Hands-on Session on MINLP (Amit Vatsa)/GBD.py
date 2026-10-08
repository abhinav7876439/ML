#!pip install gurobipy

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

tol = 1e-6

# -----------------------------
# Master problem
# min theta + y
# -----------------------------
master = gp.Model("GBD")
master.Params.OutputFlag = 0

y = master.addVar(lb=0, vtype=GRB.INTEGER, name="y")
theta = master.addVar(lb=0, name="theta")

master.setObjective(theta + y, GRB.MINIMIZE)

LB = -1e20
UB = 1e20
it = 1

points = []
cuts = []

while True:
    master.optimize()

    yk = round(y.X)
    thetak = theta.X
    LB = master.ObjVal

    print(f"Iteration {it}: Solve master -> LB = {LB:.6f}, y = {yk}, theta = {thetak:.6f}")

    # -----------------------------
    # Subproblem with y fixed
    # -----------------------------
    sub = gp.Model("subproblem")
    sub.Params.OutputFlag = 0

    x = sub.addVar(lb=0, name="x")
    sub.setObjective(x*x, GRB.MINIMIZE)

    c1 = sub.addConstr(3*x + 2*yk >= 8, name="c1")
    c2 = sub.addConstr(x + 2*yk >= 6, name="c2")

    sub.optimize()

    xk = x.X
    sub_theta = sub.ObjVal          # this is x^2 only
    UB = min(UB, sub_theta + yk)    # original objective

    lam1 = c1.Pi
    lam2 = c2.Pi

    points.append((yk, thetak))

    print(f"             Solve subproblem -> x = {xk:.6f}, UB = {UB:.6f}")
    print(f"             Multipliers: lam1 = {lam1:.6f}, lam2 = {lam2:.6f}")

    if abs(UB - LB) <= tol:
        print("STOP as LB = UB")
        break

    # GBD cut for theta only
    # theta >= xk^2 + lam1(8 - 3xk - 2y) + lam2(6 - xk - 2y)
    const_term = xk*xk + lam1*(8 - 3*xk) + lam2*(6 - xk)
    coef_y = -2*lam1 - 2*lam2

    master.addConstr(theta >= const_term + coef_y*y)
    cuts.append((const_term, coef_y))

    print(f"             Cut added: theta >= {const_term:.6f} {coef_y:+.6f} y\n")

    it += 1
# -----------------------------
# Plot
# -----------------------------
yvals = np.linspace(0, 6, 200)

plt.figure(figsize=(8, 5))

# GBD cuts: darker and thicker
for a, b in cuts:
    plt.plot(yvals, a + b*yvals, ":", linewidth=1.8, color="black")

# a few objective level curves: theta + y = c
for c in [4, 8, 16, 24]:
    plt.plot(yvals, c - yvals, "--", linewidth=0.8, color="gray")

# master points
py = [p[0] for p in points]
ptheta = [p[1] for p in points]
plt.scatter(py, ptheta, color="red", label="Master solutions", zorder=5)

for i, (yy, tt) in enumerate(points):
    plt.text(yy + 0.05, tt + 0.15, str(i+1))

# optimal point
yopt = 3
thetaopt = (2/3)**2
plt.scatter([yopt], [thetaopt], color="green", marker="*", s=150, label="Optimal point", zorder=6)

# optimal objective line
copt = yopt + thetaopt
plt.plot(yvals, copt - yvals, "-", linewidth=1.2, color="blue", label=r"Optimal line: $\theta + y = 31/9$")

plt.xlim(0, 6)
plt.ylim(0, 36)
plt.xlabel("y")
plt.ylabel(r"$\theta$")
plt.title("Generalized Benders Decomposition")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()