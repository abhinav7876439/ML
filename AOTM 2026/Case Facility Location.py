"""
AmazingDeal.com - Locating Fulfillment Centres (FFCs) with Pyomo
Every one of the 486 cities is a candidate site. Two service scenarios are compared:

    NDD = Next Day Delivery  -> a city is covered if an open FFC is within 400 km
    SDD = Same Day Delivery  -> a city is covered if an open FFC is within 200 km

  M1 Maximal Covering : open exactly P FFCs, maximise covered demand        (run per scenario)
  M2 Set Covering     : fewest FFCs so that EVERY city is covered           (run per scenario)
  M3 p-Median         : open P FFCs, minimise demand-weighted distance      (radius-free)
  M4 p-Center         : open P FFCs, minimise the WORST city-to-FFC distance (radius-free)

M3 and M4 do not use a coverage radius in their objective, so they are solved once and the
resulting network is then evaluated against BOTH the 400 km and 200 km radii.

Data: Case_Data.xlsx (sheet 1 = city / tier / demand, sheet 2 = 486x486 road distance matrix, km)
"""
import time
import numpy as np
import pandas as pd
from pyomo.environ import *

# pip install pyomo openpyxl highspy      (and/or have cplex, gurobi, scip, cbc installed)

# ----------------------------------------------------------------- settings
DATA_FILE = "Case_Data.xlsx"
P = 10                                     # number of FFCs for M1, M3, M4
RADII = {"NDD (400 km)": 400.0,            # next-day by road
         "SDD (200 km)": 200.0}            # same-day by road
UNIT_COST = 1.0                            # M3: cost per demand-unit per km (not in case data)
PMEDIAN_REQUIRE_COVERAGE = False           # M3: True -> each city may only be assigned within the radius
                                           #     (then M3 is run once per scenario; may be infeasible if P is small)
RUN_MODELS = ["M1", "M2", "M3", "M4"]
SOLVERS = ["cplex", "gurobi", "scip", "cbc", "highs"]   # tried in this order
COMPARE_ALL_SOLVERS = False                # False: use the first installed solver only
TIME_LIMIT, MIP_GAP = 300, 0.0
PRINT_COVERED_LIST = True                  # print the full list of covered cities

# --------------------------------------------------------------------- data
cities = pd.read_excel(DATA_FILE, sheet_name=0).iloc[:, :4]
cities.columns = ["id", "name", "tier", "demand"]
dist = pd.read_excel(DATA_FILE, sheet_name=1, header=None).iloc[1:487, 1:487].values.astype(float)
n = len(cities)
demand = cities["demand"].values.astype(float)
tier = cities["tier"].values
names = cities["name"].values
total_demand = demand.sum()


def coverage_sets(radius):
    """N[i] = sites j with dist[i,j] <= radius (city i is covered if any of them is open)."""
    return [np.where(dist[i] <= radius)[0].tolist() for i in range(n)]


# ------------------------------------------------------------------- models
def m1_max_cover(N, p):
    m = ConcreteModel()
    m.y = Var(range(n), within=Binary)                       # FFC opened at j
    m.x = Var(range(n), within=Binary)                       # city i covered
    m.cover = Constraint(range(n), rule=lambda m, i: m.x[i] <= quicksum(m.y[j] for j in N[i]))
    m.budget = Constraint(expr=quicksum(m.y[j] for j in range(n)) == p)
    m.obj = Objective(expr=quicksum(demand[i] * m.x[i] for i in range(n)), sense=maximize)
    return m


def m2_set_cover(N):
    m = ConcreteModel()
    m.y = Var(range(n), within=Binary)
    m.cover = Constraint(range(n), rule=lambda m, i: quicksum(m.y[j] for j in N[i]) >= 1)
    m.obj = Objective(expr=quicksum(m.y[j] for j in range(n)), sense=minimize)
    return m


def m3_p_median(p, N=None):
    """z[i,j] is continuous: at the optimum each city picks its nearest open FFC, so z is 0/1 anyway."""
    m = ConcreteModel()
    m.y = Var(range(n), within=Binary)
    allowed = [list(range(n)) for _ in range(n)] if N is None else N
    idx = [(i, j) for i in range(n) for j in allowed[i]]
    m.idx = Set(dimen=2, initialize=idx)
    m.z = Var(m.idx, within=NonNegativeReals, bounds=(0, 1))
    m.assign = Constraint(range(n), rule=lambda m, i: quicksum(m.z[i, j] for j in allowed[i]) == 1)
    m.link = Constraint(m.idx, rule=lambda m, i, j: m.z[i, j] <= m.y[j])
    m.budget = Constraint(expr=quicksum(m.y[j] for j in range(n)) == p)
    m.obj = Objective(expr=quicksum(UNIT_COST * demand[i] * dist[i, j] * m.z[i, j] for (i, j) in idx),
                      sense=minimize)
    return m


# p-Center -------------------------------------------------------------------------------------
# Textbook MIP:  min Z  s.t.  Z >= sum_j d_ij x_ij,  sum_j x_ij = 1,  x_ij <= y_j,  sum_j y_j = p.
# Its LP relaxation is very weak (slow for 486 cities), so we solve it EXACTLY another way:
# "Z <= R" is the same as "every city has an open FFC within R km" = a set-covering problem.
# So we binary-search over the sorted distinct distances for the smallest R whose minimum set
# cover needs <= P FFCs. That R is the optimal p-center radius.
def m4_p_center(p, solver_name):
    radii = np.unique(dist)                                   # sorted distinct distances
    lo, hi = 0, len(radii) - 1                                # radii[hi] (= max distance) always feasible
    best_sites, best_R, calls, proven = None, radii[hi], 0, True
    while lo < hi:
        mid = (lo + hi) // 2
        m = m2_set_cover(coverage_sets(radii[mid]))
        out = run(m, solver_name)
        calls += 1
        k = int(round(out["obj"])) if not np.isnan(out["obj"]) else n + 1
        if out["status"] != "optimal":
            proven = False
        if k <= p:
            hi, best_R, best_sites = mid, radii[mid], get_sites(m)
        else:
            lo = mid + 1
    if best_sites is None:                                    # the loop never tested radii[hi]
        m = m2_set_cover(coverage_sets(radii[hi]))
        run(m, solver_name)
        best_sites, best_R = get_sites(m), radii[hi]
    return best_R, best_sites, calls, proven


# ------------------------------------------------------------------ solving
def make_solver(name):
    """Return a configured Pyomo solver, or None if it is not installed."""
    opts = {"cplex":  {"timelimit": TIME_LIMIT, "mipgap": MIP_GAP},
            "gurobi": {"TimeLimit": TIME_LIMIT, "MIPGap": MIP_GAP},
            "scip":   {"limits/time": TIME_LIMIT, "limits/gap": MIP_GAP},
            "cbc":    {"seconds": TIME_LIMIT, "ratioGap": MIP_GAP},
            "highs":  {"time_limit": TIME_LIMIT, "mip_rel_gap": MIP_GAP}}[name]
    try:
        s = SolverFactory(name)
        if s.available(exception_flag=False):
            for k, v in opts.items():
                s.options[k] = v
            return s
    except Exception:
        pass
    return None


def installed_solvers():
    return [s for s in SOLVERS if make_solver(s) is not None]


def run(model, solver_name):
    s = make_solver(solver_name)
    t0 = time.perf_counter()
    res = s.solve(model, tee=False)
    wall = time.perf_counter() - t0
    tc = res.solver.termination_condition
    ok = tc in (TerminationCondition.optimal, TerminationCondition.maxTimeLimit, TerminationCondition.feasible)
    return {"solver": solver_name, "status": str(tc),
            "obj": value(model.obj) if ok else np.nan, "time_s": round(wall, 2)}


def get_sites(model):
    return [j for j in range(n) if value(model.y[j]) > 0.5]


# ---------------------------------------------------------------- reporting
def evaluate(sites, radius):
    """Coverage of a set of open FFCs under a radius, plus nearest-FFC assignment."""
    sites = list(sites)
    d = dist[:, sites]
    nearest = np.array(sites)[d.argmin(axis=1)]
    mind = d.min(axis=1)
    covered = np.where(mind <= radius)[0]
    uncovered = np.where(mind > radius)[0]
    return dict(covered=covered, uncovered=uncovered, nearest=nearest, mind=mind)


def report_network(sites, scenarios):
    """Print open FFCs, and for each scenario: cities covered, % demand, tier split, city list."""
    print(f"   Open FFCs ({len(sites)}): {', '.join(names[j] for j in sites)}")
    for sname, R in scenarios.items():
        ev = evaluate(sites, R)
        cov, unc = ev["covered"], ev["uncovered"]
        print(f"\n   -> {sname}: covered {len(cov)} of {n} cities "
              f"| {demand[cov].sum() / total_demand:.2%} of total demand "
              f"| not covered: {len(unc)}")
        for t in ["Metros", "Tier 1", "Tier 2", "Tier 3"]:
            tot = (tier == t).sum()
            if tot == 0:
                continue
            c = [i for i in cov if tier[i] == t]
            print(f"        {t:7s}: {len(c):3d}/{tot:3d} cities, "
                  f"{demand[c].sum() / demand[tier == t].sum():6.1%} of that tier's demand")
        if PRINT_COVERED_LIST:
            # group covered cities under the FFC that would serve them (nearest open FFC)
            for j in sites:
                served = [i for i in cov if ev["nearest"][i] == j]
                print(f"        FFC {names[j]}  serves {len(served)} cities: "
                      f"{', '.join(names[i] for i in served)}")
            if len(unc):
                print(f"        NOT covered ({len(unc)}): {', '.join(names[i] for i in unc)}")
    return


def collect_rows(model_label, scenario_label, sites, objective, scenarios):
    """One summary row per (model, scenario evaluated)."""
    rows = []
    for sname, R in scenarios.items():
        ev = evaluate(sites, R)
        rows.append({"model": model_label, "scenario_solved_for": scenario_label,
                     "evaluated_under": sname, "objective": objective, "n_open": len(sites),
                     "cities_covered": len(ev["covered"]),
                     "pct_demand_covered": round(demand[ev["covered"]].sum() / total_demand * 100, 2),
                     "open_cities": "; ".join(names[j] for j in sites)})
    return rows


def solve_first_available(build):
    """Build + solve with first installed solver (or all of them if COMPARE_ALL_SOLVERS). Returns last model/out."""
    last = None
    for sname in installed_solvers():
        m = build()
        out = run(m, sname)
        print(f"   [{sname}] {out['status']}  obj={out['obj']:.4f}  time={out['time_s']}s")
        last = (m, out)
        if not COMPARE_ALL_SOLVERS:
            break
    return last


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    avail = installed_solvers()
    if not avail:
        raise SystemExit("No MIP solver found. Try: pip install highspy")
    print(f"Solvers available: {avail} | P = {P} | scenarios: {RADII}")

    N_sets = {s: coverage_sets(R) for s, R in RADII.items()}
    rows = []

    # ---------------- M1 : Maximal covering, one run per scenario
    if "M1" in RUN_MODELS:
        for sname, R in RADII.items():
            print("\n" + "=" * 100 + f"\nM1 MAXIMAL COVERING | open exactly {P} FFCs | {sname}")
            res = solve_first_available(lambda: m1_max_cover(N_sets[sname], P))
            if res:
                m, out = res
                sites = get_sites(m)
                print(f"   OBJECTIVE: covered demand = {out['obj']:,.0f}  "
                      f"({out['obj'] / total_demand:.2%} of total demand)")
                print(f"   Given {P} FFCs, {len(evaluate(sites, R)['covered'])} cities are covered under {sname}.")
                report_network(sites, {sname: R})
                rows += collect_rows("M1 max-cover", sname, sites, out["obj"], {sname: R})

    # ---------------- M2 : Set covering, one run per scenario
    if "M2" in RUN_MODELS:
        for sname, R in RADII.items():
            print("\n" + "=" * 100 + f"\nM2 SET COVERING | cover ALL {n} cities | {sname}")
            res = solve_first_available(lambda: m2_set_cover(N_sets[sname]))
            if res:
                m, out = res
                sites = get_sites(m)
                print(f"   OBJECTIVE: minimum number of FFCs needed = {int(round(out['obj']))}")
                report_network(sites, {sname: R})
                rows += collect_rows("M2 set-cover", sname, sites, out["obj"], {sname: R})

    # ---------------- M3 : p-Median
    if "M3" in RUN_MODELS:
        if PMEDIAN_REQUIRE_COVERAGE:
            for sname, R in RADII.items():
                print("\n" + "=" * 100 + f"\nM3 p-MEDIAN | {P} FFCs | assignments restricted to {sname}")
                res = solve_first_available(lambda: m3_p_median(P, N_sets[sname]))
                if res:
                    m, out = res
                    sites = get_sites(m)
                    print(f"   OBJECTIVE: demand-weighted distance = {out['obj']:,.0f} "
                          f"(avg {out['obj'] / total_demand / UNIT_COST:.1f} km per demand unit)")
                    report_network(sites, RADII)
                    rows += collect_rows(f"M3 p-median (within {sname})", sname, sites, out["obj"], RADII)
        else:
            print("\n" + "=" * 100 + f"\nM3 p-MEDIAN | {P} FFCs | no radius in the model; coverage evaluated for NDD and SDD")
            res = solve_first_available(lambda: m3_p_median(P, None))
            if res:
                m, out = res
                sites = get_sites(m)
                print(f"   OBJECTIVE: demand-weighted distance = {out['obj']:,.0f} "
                      f"(avg {out['obj'] / total_demand / UNIT_COST:.1f} km per demand unit)")
                report_network(sites, RADII)
                rows += collect_rows("M3 p-median", "none (radius-free)", sites, out["obj"], RADII)

    # ---------------- M4 : p-Center
    if "M4" in RUN_MODELS:
        print("\n" + "=" * 100 + f"\nM4 p-CENTER | {P} FFCs | minimise the longest city-to-FFC distance")
        t0 = time.perf_counter()
        Rstar, sites, calls, proven = m4_p_center(P, avail[0])
        print(f"   [{avail[0]}] {calls} set-cover solves, {time.perf_counter() - t0:.1f}s"
              f"{'' if proven else '  (WARNING: some solves hit the time limit - result may not be optimal)'}")
        print(f"   OBJECTIVE: minimised worst-case distance = {Rstar:,.1f} km")
        print(f"   -> With {P} FFCs, EVERY city is within {Rstar:,.1f} km of an FFC.  "
              f"NDD (400 km) for all cities possible: {'YES' if Rstar <= 400 else 'NO'} | "
              f"SDD (200 km) for all cities possible: {'YES' if Rstar <= 200 else 'NO'}")
        report_network(sites, RADII)
        rows += collect_rows("M4 p-center", "none (radius-free)", sites, Rstar, RADII)

    # ------------------------------------------------------------- summary
    summary = pd.DataFrame(rows)
    show = summary.drop(columns=["open_cities"])
    print("\n\n" + "=" * 100 + "\nSUMMARY  (objective: M1 covered demand | M2 #FFCs | M3 weighted km | M4 max km)\n")
    print(show.to_string(index=False))
    summary.to_csv("ffc_results_summary.csv", index=False)
    print("\nSaved: ffc_results_summary.csv (includes the list of open cities for every run)")