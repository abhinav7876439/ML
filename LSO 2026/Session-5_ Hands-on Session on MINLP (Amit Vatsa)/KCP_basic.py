#!pip install gurobipy

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

tol = 0.001
max_iters = 4

# master problem
m = gp.Model()
m.Params.OutputFlag = 0

x = m.addVar(lb=0, ub=5, name="x")
m.setObjective(x, GRB.MAXIMIZE)

points = []
cuts = []

for k in range(max_iters):
    m.optimize()
    xk = x.X
    violation = xk**2 - 3

    print(f"Iteration {k}: x = {xk:.4f}, violation = {violation:.4f}")
    points.append(xk)

    if violation <= tol:
        break

    # tangent cut at xk:
    # 2*xk*x <= xk^2 + 3
    a = 2 * xk
    b = xk**2 + 3
    m.addConstr(a * x <= b)
    cuts.append(xk)

# plot
xx = np.linspace(0, 5, 400)
yy = xx**2

plt.figure(figsize=(8, 5))
plt.plot(xx, yy, label=r"$x^2$", linewidth=2)
plt.axhline(3, color="black", linestyle="--", label=r"$x^2 = 3$")

# tangent cuts
for xk in cuts:
    ytan = xk**2 + 2*xk*(xx - xk)
    plt.plot(xx, ytan, ":", linewidth=1)

# iterates
plt.scatter(points, [p**2 for p in points], color="red", label="iterations")
for i, p in enumerate(points):
    plt.text(p + 0.05, p**2 + 0.2, str(i))

# optimal point
xstar = np.sqrt(3)
plt.scatter([xstar], [3], color="green", marker="*", s=150, label="optimal")

plt.xlim(0, 5)
plt.ylim(0, 26)
plt.xlabel("x")
plt.ylabel("value")
plt.title("Kelley's Cutting Plane Method")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()