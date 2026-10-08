"""
Linear vs Integer Variables

Production Planning with Linear (LP) vs Integer (MIP) Variables — Pyomo + (SCIP, Gurobi)

    LP  -> NonNegativeReals  (fractional units allowed)
    MIP -> Integers          (whole units only)
"""

from pyomo.environ import (
    ConcreteModel, Var, Objective, ConstraintList,
    NonNegativeReals, Integers, maximize, SolverFactory, value,
)


# =================================================================
# LP MODEL — continuous variables
# =================================================================
model_lp = ConcreteModel()

model_lp.x = Var([1, 2], within=NonNegativeReals, bounds=(0, None))

model_lp.obj = Objective(expr=2 * model_lp.x[1] + 3 * model_lp.x[2],
                         sense=maximize)

model_lp.cons = ConstraintList()
model_lp.cons.add(model_lp.x[1] <= 4000)                    # demand P1
model_lp.cons.add(model_lp.x[2] <= 3500)                    # demand P2
model_lp.cons.add(model_lp.x[1] + model_lp.x[2] <= 5000)    # resource A
model_lp.cons.add(model_lp.x[1] + 2 * model_lp.x[2] <= 8000)  # resource B


# =================================================================
# MIP MODEL — integer variables
#    (identical to the LP above EXCEPT the domain)
# =================================================================
model_ip = ConcreteModel()

model_ip.x = Var([1, 2], within=Integers, bounds=(0, None))     

model_ip.obj = Objective(expr=2 * model_ip.x[1] + 3 * model_ip.x[2],
                         sense=maximize)

model_ip.cons = ConstraintList()
model_ip.cons.add(model_ip.x[1] <= 4000)
model_ip.cons.add(model_ip.x[2] <= 3500)
model_ip.cons.add(model_ip.x[1] + model_ip.x[2] <= 5000)
model_ip.cons.add(model_ip.x[1] + 2 * model_ip.x[2] <= 8000)


# =================================================================
# Solve
# =================================================================
solver = SolverFactory("gurobi")   # or "scip", "cbc", "glpk"
print("Solver:", solver.name, "| available:", solver.available())

solver.solve(model_lp, tee=False)
solver.solve(model_ip, tee=False)


# =================================================================
# Output
# =================================================================
print("\n" + "=" * 50)
print(f"{'':15s}{'LP':>16s}{'MIP':>16s}")
print("=" * 50)
print(f"{'x1':15s}{value(model_lp.x[1]):>16.4f}{value(model_ip.x[1]):>16.0f}")
print(f"{'x2':15s}{value(model_lp.x[2]):>16.4f}{value(model_ip.x[2]):>16.0f}")
print(f"{'Objective':15s}{value(model_lp.obj):>16.4f}{value(model_ip.obj):>16.4f}")

# ---- Slacks ----
print("\nConstraint slacks:")
print(f"  {'':10s}{'LP':>14s}{'MIP':>14s}")

lp_s1 = value(model_lp.cons[1].upper) - value(model_lp.cons[1].body)
lp_s2 = value(model_lp.cons[2].upper) - value(model_lp.cons[2].body)
lp_s3 = value(model_lp.cons[3].upper) - value(model_lp.cons[3].body)
lp_s4 = value(model_lp.cons[4].upper) - value(model_lp.cons[4].body)

ip_s1 = value(model_ip.cons[1].upper) - value(model_ip.cons[1].body)
ip_s2 = value(model_ip.cons[2].upper) - value(model_ip.cons[2].body)
ip_s3 = value(model_ip.cons[3].upper) - value(model_ip.cons[3].body)
ip_s4 = value(model_ip.cons[4].upper) - value(model_ip.cons[4].body)

print(f"  {'cons[1]':10s}{lp_s1:>14.2f}{ip_s1:>14.2f}")
print(f"  {'cons[2]':10s}{lp_s2:>14.2f}{ip_s2:>14.2f}")
print(f"  {'cons[3]':10s}{lp_s3:>14.2f}{ip_s3:>14.2f}")
print(f"  {'cons[4]':10s}{lp_s4:>14.2f}{ip_s4:>14.2f}")

# ---- Integrality comment ----
lp_x1 = value(model_lp.x[1])
lp_x2 = value(model_lp.x[2])
if abs(lp_x1 - round(lp_x1)) > 1e-6 or abs(lp_x2 - round(lp_x2)) > 1e-6:
    gap = value(model_lp.obj) - value(model_ip.obj)
    print(f"\n>> LP optimum is FRACTIONAL — MIP differs.")
    print(f">> Integrality gap (LP upper bound - MIP) = {gap:.4f}")
else:
    print("\n>> LP optimum is already integral — MIP matches LP.")