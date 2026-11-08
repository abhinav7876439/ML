import itertools
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


D = [
    [0,   122, 272, 391, 212, 154],
    [122, 0,   261, 364, 128, 269],
    [272, 261, 0,   188, 372, 321],
    [391, 364, 188, 0,   391, 375],
    [212, 128, 372, 391, 0,   286],
    [154, 269, 321, 375, 286, 0  ],
]

num_city = len(D)

model = pyo.ConcreteModel()

model.x = pyo.Var(
    [(i, j) for i in range(num_city) for j in range(num_city) if i != j],
    within=pyo.Binary,
)

model.obj = pyo.Objective(
    expr=sum(D[i][j] * model.x[i, j]
             for i in range(num_city)
             for j in range(num_city) if i != j),
    sense=pyo.minimize,
)

model.out_deg = pyo.ConstraintList()
for i in range(num_city):
    model.out_deg.add(sum(model.x[i, j] for j in range(num_city) if j != i) == 1)

model.in_deg = pyo.ConstraintList()
for j in range(num_city):
    model.in_deg.add(sum(model.x[i, j] for i in range(num_city) if i != j) == 1)

# Empty DFJ list — we add cuts manually below
model.dfj = pyo.ConstraintList()

solver = SolverFactory('cplex')
solver.solve(model, tee=False)

# ==============================================================================
# Q1. Run the code with NO DFJ cuts.
#     What subtours appear? What is the objective?
# ==============================================================================

# print("\n" + "=" * 50)
# print(f"[Q1] Objective: {pyo.value(model.obj):.0f} km")

# arcs = [(i, j) for (i, j) in model.x if pyo.value(model.x[i, j]) > 0.5]
# print("[Q1] Selected arcs:", arcs)

# succ = {i: j for (i, j) in arcs}
# visited = set()
# all_cycles = []
# for start in range(num_city):
#     if start in visited:
#         continue
#     cycle = []
#     cur = start
#     while cur not in visited:
#         visited.add(cur)
#         cycle.append(cur)
#         cur = succ[cur]
#     all_cycles.append(cycle)
# print("[Q1] Cycles found:", all_cycles)


# ==============================================================================
# Q2. Add ONE DFJ constraint to kill the subtour
# ==============================================================================

# Cut: leave S = {0, 5}
# model.dfj.add(
#     model.x[0, 1] + model.x[0, 2] + model.x[0, 3] + model.x[0, 4]
#   + model.x[5, 1] + model.x[5, 2] + model.x[5, 3] + model.x[5, 4] >= 1
# )

# solver.solve(model, tee=False)

# print("\n" + "=" * 50)
# print(f"[Q2] Objective: {pyo.value(model.obj):.0f} km")

# arcs = [(i, j) for (i, j) in model.x if pyo.value(model.x[i, j]) > 0.5]
# print("[Q2] Selected arcs:", arcs)

# succ = {i: j for (i, j) in arcs}
# visited = set()
# all_cycles = []
# for start in range(num_city):
#     if start in visited:
#         continue
#     cycle = []
#     cur = start
#     while cur not in visited:
#         visited.add(cur)
#         cycle.append(cur)
#         cur = succ[cur]
#     all_cycles.append(cycle)
# print("[Q2] Cycles found:", all_cycles)

# ==============================================================================
# # Cut: leave S = {1, 4}
# model.dfj.add(
#     model.x[1, 0] + model.x[1, 2] + model.x[1, 3] + model.x[1, 5]
#   + model.x[4, 0] + model.x[4, 2] + model.x[4, 3] + model.x[4, 5] >= 1
# )

# solver.solve(model, tee=False)

# print("\n" + "=" * 50)
# print(f"[Q2] Objective: {pyo.value(model.obj):.0f} km")

# arcs = [(i, j) for (i, j) in model.x if pyo.value(model.x[i, j]) > 0.5]
# print("[Q2] Selected arcs:", arcs)

# succ = {i: j for (i, j) in arcs}
# visited = set()
# all_cycles = []
# for start in range(num_city):
#     if start in visited:
#         continue
#     cycle = []
#     cur = start
#     while cur not in visited:
#         visited.add(cur)
#         cycle.append(cur)
#         cur = succ[cur]
#     all_cycles.append(cycle)
# print("[Q2] Cycles found:", all_cycles)
# ==============================================================================
# Cut: leave S = {2,3}
# model.dfj.add(
#     model.x[2, 0] + model.x[2, 1] + model.x[2, 4] + model.x[2, 5]
#   + model.x[3, 0] + model.x[3, 1] + model.x[3, 4] + model.x[3, 5] >= 1
# )

# solver.solve(model, tee=False)

# print("\n" + "=" * 50)
# print(f"[Q2] Objective: {pyo.value(model.obj):.0f} km")

# arcs = [(i, j) for (i, j) in model.x if pyo.value(model.x[i, j]) > 0.5]
# print("[Q2] Selected arcs:", arcs)

# succ = {i: j for (i, j) in arcs}
# visited = set()
# all_cycles = []
# for start in range(num_city):
#     if start in visited:
#         continue
#     cycle = []
#     cur = start
#     while cur not in visited:
#         visited.add(cur)
#         cycle.append(cur)
#         cur = succ[cur]
#     all_cycles.append(cycle)
# print("[Q2] Cycles found:", all_cycles)
# ==============================================================================

# ==============================================================================
# Q3. Add a DFJ cut to kill the remaining subtour {2, 3}:
# ==============================================================================

# Cut: leave S = {2, 3}
# model.dfj.add(
#     model.x[2, 0] + model.x[2, 1] + model.x[2, 4] + model.x[2, 5]
#   + model.x[3, 0] + model.x[3, 1] + model.x[3, 4] + model.x[3, 5] >= 1
# )

# solver.solve(model, tee=False)

# print("\n" + "=" * 50)
# print(f"[Q3] Objective: {pyo.value(model.obj):.0f} km")

# arcs = [(i, j) for (i, j) in model.x if pyo.value(model.x[i, j]) > 0.5]
# print("[Q3] Selected arcs:", arcs)

# succ = {i: j for (i, j) in arcs}
# visited = set()
# all_cycles = []
# for start in range(num_city):
#     if start in visited:
#         continue
#     cycle = []
#     cur = start
#     while cur not in visited:
#         visited.add(cur)
#         cycle.append(cur)
#         cur = succ[cur]
#     all_cycles.append(cycle)
# print("[Q3] Cycles found:", all_cycles)


# # ==============================================================================
# # Q4. Instead of cutting {2, 3} in Q3, try cutting {1, 4} as the SECOND cut.
# # ==============================================================================

# # # Cut: leave S = {1, 4}
# model.dfj.add(
#     model.x[1, 0] + model.x[1, 2] + model.x[1, 3] + model.x[1, 5]
#   + model.x[4, 0] + model.x[4, 2] + model.x[4, 3] + model.x[4, 5] >= 1
# )

# solver.solve(model, tee=False)

# print("\n" + "=" * 50)
# print(f"[Q4] Objective: {pyo.value(model.obj):.0f} km")

# arcs = [(i, j) for (i, j) in model.x if pyo.value(model.x[i, j]) > 0.5]
# print("[Q4] Selected arcs:", arcs)

# succ = {i: j for (i, j) in arcs}
# visited = set()
# all_cycles = []
# for start in range(num_city):
#     if start in visited:
#         continue
#     cycle = []
#     cur = start
#     while cur not in visited:
#         visited.add(cur)
#         cycle.append(cur)
#         cur = succ[cur]
#     all_cycles.append(cycle)
# print("[Q4] Cycles found:", all_cycles)

# ==============================================================================

# ---------- DFJ subtour elimination constraints ----------
model.dfj = pyo.ConstraintList()
all_cities = set(range(num_city))
for size in range(2, num_city):
    for S in itertools.combinations(range(num_city), size):
        S = set(S)
        complement = all_cities - S
        model.dfj.add(
            sum(model.x[i, j]
                for i in S for j in complement
                if i != j) >= 1
        )


# ==============================================================================
# 2 <= |S| <= n-1
# model.dfj = pyo.ConstraintList()
# for size in range(2, num_city):                         
#     for S in itertools.combinations(range(num_city), size):
#         S = set(S)
#         model.dfj.add(
#             sum(model.x[i, j]
#                 for i in S for j in S
#                 if i != j) <= size - 1
#         )

# ==============================================================================


# # ---------- Solve with CPLEX ----------
solver = SolverFactory('cplex')
results = solver.solve(model, tee=True)

# ---------- Extract solution ----------
print("\n" + "=" * 50)
print(f"Objective (total distance): {pyo.value(model.obj):.0f} km")

# ==============================================================================

print("\n" + "=" * 50)
print("Tour walker on the CURRENT model (after Q1–Q3 cuts):")

arcs = [(i, j) for (i, j) in model.x if pyo.value(model.x[i, j]) > 0.5]

tour = [0]
current = 0
seen = {0}
for _ in range(num_city - 1):
    nxt = None
    for (i, j) in arcs:
        if i == current and j not in seen:
            nxt = j
            break
    if nxt is None:
        print(f"  ERROR: No unused arc leaves city {current}.")
        print(f"  Tour so far: {tour}")
        print("  -> SUBTOURS still present (not all DFJ cuts added).")
        break
    tour.append(nxt)
    seen.add(nxt)
    current = nxt
else:
    tour.append(0)
    print("  Tour:", " -> ".join(map(str, tour)))