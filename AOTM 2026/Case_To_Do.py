"""
AmazingDeal.com - Locating Fulfillment Centres (FFCs) with Pyomo

Fill in every  ____  (marked TODO) so that the script runs.

No FFC is assumed to exist: every one of the 486 cities is a candidate site.
No tier quotas: tiers are used only for reporting.

  Problem 1 : Open exactly P = 25 FFCs and cover as much demand as possible.
              A city is covered if an open FFC is within the radius
              (NDD = 400 km, SDD = 200 km, solved separately).      -> Maximal Covering
  Problem 2 : Minimum number of FFCs such that EVERY city is covered. -> Set Covering

Data: Case_Data.xlsx (sheet 1 = city/tier/demand, sheet 2 = 486x486 road distance matrix, km)

Notation
  i = demand city, j = candidate site (both range over the 486 cities)
  N[i] = set of sites j that are within the radius of city i
  y[j] = 1 if an FFC is opened in city j           (binary)
  x[i] = 1 if city i is covered                     (binary, Problem 1 only)
"""

# ----------------------------------------------------------------------------
# DISCUSSION QUESTIONS (after the code runs)
#  a) How many FFCs does Problem 2 need for NDD vs SDD? Why the big difference?
#  b) In Problem 1, which tiers are under-covered with SDD? Is that a problem?
#  c) What happens to the % covered if you change P to 15 or 35?
# ----------------------------------------------------------------------------

import time
import numpy as np
import pandas as pd
from pyomo.environ import *

# ----------------------------------------------------------------- settings
DATA_FILE = "Case_Data.xlsx"
P = ____                                                # TODO 1: number of FFCs for Problem 1
RADII = {"NDD (400 km)": 400}                           # coverage radius in km
RUN_PROBLEMS = ["P1", "P2"]                             # choose which problems to run
SOLVERS = ["highs", "cbc", "scip", "gurobi", "cplex"]   # whichever are installed
TIME_LIMIT, MIP_GAP = 300, 0.0

# --------------------------------------------------------------------- data
cities = pd.read_excel(DATA_FILE, sheet_name=0).iloc[:, :4]
cities.columns = ["id", "name", "tier", "demand"]
dist = pd.read_excel(DATA_FILE, sheet_name=1, header=None).iloc[1:487, 1:487].values.astype(float)
n = len(cities)
demand = cities["demand"].values
tier = cities["tier"].values
total_demand = demand.sum()


def coverage_sets(radius):
    """N[i] = list of sites j with dist[i, j] <= radius
    (city i is covered if at least one site in N[i] is open).
    Hint: np.where(condition)[0].tolist() returns the indices where condition is True."""
    # TODO 2: return a list with one entry per city i
    return [np.where(____)[0].tolist() for i in range(n)]


# ------------------------------------------------------------------- models
def problem1_max_cover(N, p):
    """Maximal covering: open exactly p FFCs, maximise covered demand."""
    m = ConcreteModel()
    # TODO 3: decision variables (both binary)
    m.y = Var(range(n), within=____)         # y[j] = 1 if an FFC is opened in city j
    m.x = Var(range(n), within=____)         # x[i] = 1 if city i is covered

    # TODO 4: city i can only be covered if at least one FFC in N[i] is open
    #         x[i] <= sum of y[j] for j in N[i]
    m.cover = Constraint(range(n), rule=lambda m, i: m.x[i] ____ quicksum(m.y[j] for j in ____))

    # TODO 5: open exactly p FFCs
    m.budget = Constraint(expr=quicksum(m.y[j] for j in range(n)) ____ p)

    # TODO 6: maximise the covered demand  (sum of demand[i] * x[i])
    m.obj = Objective(expr=quicksum(____ * m.x[i] for i in range(n)), sense=____)
    return m


def problem2_set_cover(N):
    """Set covering: minimum number of FFCs so that every city is covered."""
    m = ConcreteModel()
    m.y = Var(range(n), within=Binary)

    # TODO 7: every city needs AT LEAST ONE open FFC within its radius
    m.cover = Constraint(range(n), rule=lambda m, i: quicksum(m.y[j] for j in N[i]) ____ 1)

    # TODO 8: minimise the number of FFCs opened
    m.obj = Objective(expr=quicksum(____ for j in range(n)), sense=____)
    return m


# ------------------------------------------------------------------ solving
def make_solver(name):
    """Return a configured Pyomo solver, or None if it is not installed."""
    opts_by_solver = {
        "cplex":  {"timelimit": TIME_LIMIT, "mipgap": MIP_GAP},
        "gurobi": {"TimeLimit": TIME_LIMIT, "MIPGap": MIP_GAP},
        "scip":   {"limits/time": TIME_LIMIT, "limits/gap": MIP_GAP},
        "cbc":    {"seconds": TIME_LIMIT, "ratioGap": MIP_GAP},
        "highs":  {"time_limit": TIME_LIMIT, "mip_rel_gap": MIP_GAP},
    }
    try:
        s = SolverFactory(name)
        if s.available(exception_flag=False):
            for k, v in opts_by_solver[name].items():
                s.options[k] = v
            return s
    except Exception:
        pass
    return None


def run(model, solver_name):
    s = make_solver(solver_name)
    if s is None:
        return None
    t0 = time.perf_counter()
    res = s.solve(model, tee=False)
    wall = time.perf_counter() - t0
    ok = res.solver.termination_condition in (TerminationCondition.optimal,
                                              TerminationCondition.maxTimeLimit,
                                              TerminationCondition.feasible)
    return {"solver": solver_name, "status": str(res.solver.termination_condition),
            "obj": value(model.obj) if ok else np.nan, "time_s": round(wall, 3)}


def report_sites(model, N):
    """Print the chosen cities and the coverage (radius sets N), overall and by tier."""
    # TODO 9: sites = cities j whose y[j] is (numerically) 1.  Hint: value(model.y[j]) > 0.5
    sites = [j for j in range(n) if ____]
    site_set = set(sites)
    # TODO 10: city i is covered if ANY site in N[i] is in site_set
    covered = [i for i in range(n) if any(____ for j in N[i])]
    print(f"   FFC cities ({len(sites)}): {[cities.name[j] for j in sites]}")
    print(f"   covered: {len(covered)}/{n} cities | {demand[covered].sum() / total_demand:.2%} of demand")
    for t in ["Metros", "Tier 1", "Tier 2", "Tier 3"]:
        idx = [i for i in covered if tier[i] == t]
        print(f"     {t:7s}: {len(idx)}/{(tier == t).sum()} cities, "
              f"{demand[idx].sum() / demand[tier == t].sum():.1%} of that tier's demand")
    return sites


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    rows = []
    N_sets = {name: coverage_sets(r) for name, r in RADII.items()}

    jobs = []   # (label, model builder, coverage sets for reporting, objective unit)
    for rname, N in N_sets.items():
        if "P1" in RUN_PROBLEMS:
            jobs.append((f"Problem 1: max-cover p={P} | {rname}",
                         lambda N=N: problem1_max_cover(N, P), N, "pct"))
        if "P2" in RUN_PROBLEMS:
            jobs.append((f"Problem 2: set-cover | {rname}",
                         lambda N=N: problem2_set_cover(N), N, "count"))

    for label, build, N, unit in jobs:
        print("\n" + "=" * 90 + f"\n{label}")
        last = None
        for sname in SOLVERS:
            m = build()
            out = run(m, sname)
            if out is None:
                print(f"   [{sname}] not available - skipped")
                continue
            if unit == "pct":
                out["obj"] = out["obj"] / total_demand * 100      # % of total demand covered
            rows.append({"model": label, **out})
            print(f"   [{sname}] {out['status']}  obj={out['obj']:.4f}  time={out['time_s']}s")
            last = m
        if last is not None:
            report_sites(last, N)

    summary = pd.DataFrame(rows)
    print("\n\nSUMMARY  (Problem 1: % demand covered | Problem 2: number of FFCs)\n")
    print(summary.to_string(index=False))
    summary.to_csv("solver_comparison.csv", index=False)

