#!/usr/bin/env python3
import math

#input
INSTANCE_FILE = "Solomon/C101.txt"
DISTANCE_DECIMALS = 2

#reading the isntance, DO NOT CHANGE
def read_solomon(filename):
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

    n = len(customers)
    o = 0
    d = n + 1

    coord = {o: (depot[1], depot[2]), d: (depot[1], depot[2])}
    q = {o: 0.0, d: 0.0}
    a = {o: depot[4], d: depot[4]}
    b = {o: depot[5], d: depot[5]}
    s = {o: 0.0, d: 0.0}
    original_id = {o: 0, d: 0}

    for new_id, row in enumerate(customers, start=1):
        old_id, xx, yy, dem, ready, due, serv = row
        coord[new_id] = (xx, yy)
        q[new_id] = dem
        a[new_id] = ready
        b[new_id] = due
        s[new_id] = serv
        original_id[new_id] = old_id

    if K is None or Q is None:
        raise ValueError("Could not parse vehicle number/capacity.")

    return K, Q, o, d, list(range(1, n + 1)), coord, q, a, b, s, original_id

def distance(coord, i, j):
    xi, yi = coord[i]
    xj, yj = coord[j]
    val = math.hypot(xi - xj, yi - yj)
    if DISTANCE_DECIMALS is not None:
        val = round(val, DISTANCE_DECIMALS)
    return val

if __name__ == "__main__":

    # Kmax: maximum number of available vehicles
    # Q: vehicle capacity
    # o: start depot node id
    # d: end depot node id
    # C: list of customer node ids
    # coord[i]: coordinates of node i
    # q[i]: demand of node i
    # a[i], b[i]: earliest and latest service start time at node i
    # s[i]: service time at node i
    # original_id[i]: original Solomon instance id of renamed node i

    #reading the instance file
    Kmax, Q, o, d, C, coord, q, a, b, s, original_id = read_solomon(INSTANCE_FILE)
    print("Instance:", INSTANCE_FILE)
    print("Customers:", len(C))
    print("Vehicle capacity:", Q)
    print("Available vehicles:", Kmax)

    #Ex 1: print the distance between depot and customer 10


    #Ex 2: Create a sample route


    #Ex 3: check if the route created is feasible: capacity and time-windows
    #capacity


    #timewindows


    #Ex 4: greedy constructive heuristics



    #Ex 5: print solution



    #Ex 6: LS, start with the greedy soltuion as s_0



    #Ex 7: print the LS solution




