#!/usr/bin/env python3
import math
import gurobipy as gp
from gurobipy import GRB

# ===================== USER INPUT =====================
INSTANCE_FILE = "Solomon/R101.txt"   # Solomon instance file
OUTPUT_LP = "inst.lp"
TIME_LIMIT = 600
MIP_GAP = 0.00

CUSTOMER_LIMIT = 50  # use 25 or 50 for smaller tests; None means all customers
DISTANCE_DECIMALS = 2   # Solomon distances usually compared with 2 decimals

USE_VEHICLE_FIXED_COST = False
VEHICLE_FIXED_COST = 100000.0   # use this if you want lexicographic min vehicles first
# ======================================================

def read_solomon(filename, customer_limit=None):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    K, Q = None, None
    for idx, line in enumerate(lines):
        if "NUMBER" in line.upper() and "CAPACITY" in line.upper():
            for nxt in lines[idx + 1:]:
                parts = nxt.split()
                if len(parts) >= 2:
                    try:
                        K = int(float(parts[0]))
                        Q = float(parts[1])
                        break
                    except ValueError:
                        pass
            break

    rows = []
    for line in lines:
        p = line.split()
        if len(p) >= 7:
            try:
                node = int(float(p[0]))
                xcoord = float(p[1])
                ycoord = float(p[2])
                demand = float(p[3])
                ready = float(p[4])
                due = float(p[5])
                service = float(p[6])
                rows.append((node, xcoord, ycoord, demand, ready, due, service))
            except ValueError:
                pass

    rows = sorted(rows, key=lambda z: z[0])
    if not rows or rows[0][0] != 0:
        raise ValueError("Could not parse Solomon data. Depot row with customer number 0 not found.")

    depot = rows[0]
    customers = rows[1:]
    if customer_limit is not None:
        customers = customers[:customer_limit]

    # Renumber: start depot o=0, customers 1..n, end depot d=n+1
    n = len(customers)
    o = 0
    d = n + 1

    coord = {o: (depot[1], depot[2]), d: (depot[1], depot[2])}
    q = {o: 0.0, d: 0.0}
    a = {o: depot[4], d: depot[4]}
    b = {o: depot[5], d: depot[5]}
    s = {o: 0.0, d: 0.0}

    for new_id, row in enumerate(customers, start=1):
        _, xx, yy, dem, ready, due, serv = row
        coord[new_id] = (xx, yy)
        q[new_id] = dem
        a[new_id] = ready
        b[new_id] = due
        s[new_id] = serv

    if K is None or Q is None:
        raise ValueError("Could not parse vehicle number/capacity.")

    return K, Q, o, d, list(range(1, n + 1)), coord, q, a, b, s

def distance(coord, i, j):
    xi, yi = coord[i]
    xj, yj = coord[j]
    val = math.hypot(xi - xj, yi - yj)
    if DISTANCE_DECIMALS is not None:
        val = round(val, DISTANCE_DECIMALS)
    return val

def build_and_solve():
    Kmax, Q, o, d, C, coord, q, a, b, s = read_solomon(INSTANCE_FILE, CUSTOMER_LIMIT)

    K = list(range(1, Kmax + 1))
    N = [o] + C + [d]

    # Feasible directed arcs: no incoming to start depot, no outgoing from end depot
    A = []
    c = {}
    t = {}

    for i in N:
        for j in N:
            if i == j:
                continue
            if j == o:
                continue
            if i == d:
                continue
            if i == o and j == d:
                continue

            dij = distance(coord, i, j)

            # Basic time-window feasibility filter
            # If earliest possible departure from i cannot reach j by b_j, arc is useless.
            if a[i] + s[i] + dij > b[j] + 1e-9:
                continue

            A.append((i, j))
            c[i, j] = dij
            t[i, j] = dij

    print("Instance:", INSTANCE_FILE)
    print("Customers:", len(C))
    print("Vehicles:", Kmax)
    print("Capacity:", Q)
    print("Nodes:", len(N))
    print("Arcs:", len(A))
    print("Total demand:", sum(q[i] for i in C))

    model = gp.Model("VRPTW_arc_flow")

    # IMPORTANT:
    # Manually name variables so the exported LP has names like x_0_1_1.
    # GCG/SCIP can fail on Gurobi's default bracket names like x[0,1,1].
    x = {}
    for (i, j) in A:
        for k in K:
            x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")

    z = {}
    for k in K:
        z[k] = model.addVar(vtype=GRB.BINARY, name=f"z_{k}")

    T = {}
    for i in N:
        for k in K:
            T[i, k] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"T_{i}_{k}")

    model.update()

    if USE_VEHICLE_FIXED_COST:
        model.setObjective(
            gp.quicksum(c[i, j] * x[i, j, k] for (i, j) in A for k in K)
            + VEHICLE_FIXED_COST * gp.quicksum(z[k] for k in K),
            GRB.MINIMIZE
        )
    else:
        model.setObjective(
            gp.quicksum(c[i, j] * x[i, j, k] for (i, j) in A for k in K),
            GRB.MINIMIZE
        )

    # Each customer visited exactly once
    for i in C:
        model.addConstr(
            gp.quicksum(x[i, j, k] for k in K for j in N if (i, j) in A) == 1,
            name=f"visit_{i}"
        )

    # Start depot and end depot usage
    for k in K:
        model.addConstr(
            gp.quicksum(x[o, j, k] for j in N if (o, j) in A) == z[k],
            name=f"depart_{k}"
        )
        model.addConstr(
            gp.quicksum(x[i, d, k] for i in N if (i, d) in A) == z[k],
            name=f"return_{k}"
        )

    # Flow conservation for customers
    for k in K:
        for i in C:
            model.addConstr(
                gp.quicksum(x[i, j, k] for j in N if (i, j) in A)
                - gp.quicksum(x[j, i, k] for j in N if (j, i) in A)
                == 0,
                name=f"flow_{i}_{k}"
            )

    # Capacity
    for k in K:
        model.addConstr(
            gp.quicksum(
                q[i] * gp.quicksum(x[i, j, k] for j in N if (i, j) in A)
                for i in C
            ) <= Q,
            name=f"cap_{k}"
        )

    # Time windows for depots
    for k in K:
        model.addConstr(T[o, k] >= a[o] * z[k], name=f"tw_start_lb_{k}")
        model.addConstr(T[o, k] <= b[o] * z[k], name=f"tw_start_ub_{k}")
        model.addConstr(T[d, k] >= a[d] * z[k], name=f"tw_end_lb_{k}")
        model.addConstr(T[d, k] <= b[d] * z[k], name=f"tw_end_ub_{k}")

    # Time windows for customers, active only if customer is served by vehicle k
    for k in K:
        for i in C:
            visit_ik = gp.quicksum(x[i, j, k] for j in N if (i, j) in A)
            model.addConstr(T[i, k] >= a[i] * visit_ik, name=f"tw_lb_{i}_{k}")
            model.addConstr(T[i, k] <= b[i] * visit_ik, name=f"tw_ub_{i}_{k}")

    # Time propagation constraints
    for k in K:
        for (i, j) in A:
            Mij = b[i] + s[i] + t[i, j]
            model.addConstr(
                T[j, k] >= T[i, k] + s[i] + t[i, j] - Mij * (1 - x[i, j, k]),
                name=f"time_{i}_{j}_{k}"
            )

    model.Params.TimeLimit = TIME_LIMIT
    model.Params.MIPGap = MIP_GAP
    model.Params.OutputFlag = 1

    model.update()
    model.write(OUTPUT_LP)
    print("Model written to:", OUTPUT_LP)

    model.optimize()

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        if model.SolCount > 0:
            print("\nObjective:", model.ObjVal)
            used = [k for k in K if z[k].X > 0.5]
            print("Vehicles used:", len(used), used)

            for k in used:
                route = [o]
                cur = o
                while cur != d:
                    nxts = [j for j in N if (cur, j) in A and x[cur, j, k].X > 0.5]
                    if not nxts:
                        break
                    cur = nxts[0]
                    route.append(cur)
                print(f"Vehicle {k}: {route}")
                for node in route:
                    print(f"  T[{node},{k}] = {T[node,k].X:.2f}")
    else:
        print("No solution. Status:", model.Status)

if __name__ == "__main__":
    build_and_solve()
