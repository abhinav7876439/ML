#!pip install gurobipy

# https://printster.in/pricecalculator

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tol = 0.001
max_iters = 2

# master problem
m = gp.Model("kelley")
m.Params.OutputFlag = 0

x = m.addVar(lb=-3, ub=3, name="x")
y = m.addVar(lb=-3, ub=3, name="y")
m.setObjective(x + y, GRB.MAXIMIZE)

rows = []
cuts = []

for k in range(max_iters):
    m.optimize()

    xk = x.X
    yk = y.X
    gk = xk*xk + yk*yk - 4   # violation

    cut_text = ""
    print(f"Iteration {k}: x = {xk:.4f}, y = {yk:.4f}, violation = {gk:.4f}")

    if gk <= tol:
        rows.append([k, xk, yk, gk, cut_text])
        break

    # Kelley cut:
    # 2*xk*x + 2*yk*y <= xk^2 + yk^2 + 4
    a = 2*xk
    b = 2*yk
    c = xk*xk + yk*yk + 4

    m.addConstr(a*x + b*y <= c)

    cut_text = f"{a:.4f} x + {b:.4f} y <= {c:.4f}"
    print("   Cut added:", cut_text)

    rows.append([k, xk, yk, gk, cut_text])
    cuts.append((a, b, c))

df = pd.DataFrame(rows, columns=["iter", "x", "y", "violation", "cut_added"])

print("\nIteration table:")
print(df.to_string(index=False))

# plot
theta = np.linspace(0, 2*np.pi, 400)
xc = 2*np.cos(theta)
yc = 2*np.sin(theta)

plt.figure(figsize=(7,7))
plt.plot(xc, yc, label="x^2 + y^2 = 4", linewidth=2)

# box
plt.plot([-3, 3, 3, -3, -3], [-3, -3, 3, 3, -3], "k--", linewidth=1)

# cuts
xx = np.linspace(-3.2, 3.2, 400)
for a, b, c in cuts:
    if abs(b) > 1e-8:
        yy = (c - a*xx)/b
        plt.plot(xx, yy, linewidth=1)
    else:
        plt.axvline(c/a, linewidth=1)

# iteration points only, no path
plt.scatter(df["x"], df["y"], s=40, label="iteration points")
for i in range(len(df)):
    plt.text(df.loc[i, "x"] + 0.03, df.loc[i, "y"] + 0.03, str(df.loc[i, "iter"]))

plt.scatter([1.414], [1.414], marker="*", s=150, label="optimal point (1.414,1.414)")

plt.xlim(-3.2, 3.2)
plt.ylim(-3.2, 3.2)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True)
plt.legend()
plt.title("Kelley's Cutting Plane Method")
plt.xlabel("x")
plt.ylabel("y")
plt.show()