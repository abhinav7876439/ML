# --------------------------------------------------------------------------
# 0. Balance Problem
# --------------------------------------------------------------------------


# from pyomo.environ import *
# import pyomo.environ as pyo

# # -------------------------------------------------
# # 1. Input data
# # -------------------------------------------------

# edges_list = [('BOM', 'SOL'), ('BEL', 'HUB'), ('BOM', 'HUB'), ('SOL', 'GUL'), ('HUB', 'GUL'), ('HUB', 'BAN'), ('GUL', 'BAN'), ('GUL', 'CHE'), ('BAN', 'CHE')]
# supply_dict = [{'BOM': 190}, {'BEL': 110}]
# demand_dict = [{'CHE': 200}, {'BAN': 100}]
# cost = [3, 2, 6,  4, 4, 4, 2, 7, 3]
# print(edges_list,'\n', supply_dict,'\n', demand_dict,'\n', cost)

# # -------------------------------------------------
# # 2. Convert dictionaries into a convenient format
# # -------------------------------------------------

# supply = {}
# for item in supply_dict:
#     supply.update(item)

# demand = {}
# for item in demand_dict:
#     demand.update(item)

# nodes = set()
# for i, j in edges_list:
#     nodes.add(i)
#     nodes.add(j)

# # Net supply: positive for supply nodes,
# # negative for demand nodes, zero for transshipment nodes
# balance = {n: 0 for n in nodes}

# for n, quantity in supply.items():
#     balance[n] = quantity

# for n, quantity in demand.items():
#     balance[n] = -quantity

# # Associate each edge with its transportation cost
# cost_dict = {
#     edges_list[k]: cost[k]
#     for k in range(len(edges_list))
# }

# # -------------------------------------------------
# # 3. Create the optimization model
# # -------------------------------------------------

# model = pyo.ConcreteModel()

# model.N = pyo.Set(initialize=sorted(nodes))
# model.A = pyo.Set(dimen=2, initialize=edges_list)

# # Flow through each edge
# model.x = pyo.Var(model.A, domain=pyo.NonNegativeReals)

# # -------------------------------------------------
# # 4. Objective: minimize transportation cost
# # -------------------------------------------------

# model.obj = pyo.Objective(
#     expr=sum(
#         cost_dict[i, j] * model.x[i, j]
#         for i, j in edges_list
#     ),
#     sense=pyo.minimize
# )

# # -------------------------------------------------
# # 5. Flow conservation constraints
# # Outgoing flow - incoming flow = net supply
# # -------------------------------------------------

# def flow_balance_rule(model, n):
#     outgoing = sum(
#         model.x[i, j]
#         for i, j in edges_list if i == n
#     )

#     incoming = sum(
#         model.x[i, j]
#         for i, j in edges_list if j == n
#     )

#     return outgoing - incoming == balance[n]

# model.flow_balance = pyo.Constraint(
#     model.N,
#     rule=flow_balance_rule
# )


# # --------------------------------------------------------------------
# # Solapur-Gulbarga pipeline is disrupted
# model.pipeline_disruption = pyo.Constraint(
#     expr=model.x['SOL', 'GUL'] == 0
# )

# # # Bangalore can receive at most 150 million barrels
# model.bangalore_capacity = pyo.Constraint(
#     expr=model.x['HUB', 'BAN']
#     + model.x['GUL', 'BAN'] <= 150
# )

# # --------------------------------------------------------------------


# # -------------------------------------------------
# # 6. Solve the model
# # -------------------------------------------------

# solver = pyo.SolverFactory('scip')

# if not solver.available(exception_flag=False):
#     raise RuntimeError(
#         "SCIP is unavailable. Run: pip install scip"
#     )

# results = solver.solve(model, tee=False)

# # -------------------------------------------------
# # 7. Display the results
# # -------------------------------------------------


# if results.solver.termination_condition == pyo.TerminationCondition.optimal:

#     print("\nOptimal transportation cost:", pyo.value(model.obj))
#     print("\nOptimal flow on each edge:")

#     for i, j in edges_list:
#         flow = pyo.value(model.x[i, j])
#         print("{} -> {}: {:.2f}".format(i, j, flow))

# else:
#     print("Solver status:", results.solver.status)
#     print("Termination condition:",
#           results.solver.termination_condition)

# --------------------------------------------------------------------------
# 0. Supply is more than demand
# --------------------------------------------------------------------------

import pyomo.environ as pyo

# # -------------------------------------------------
# # 1. Input data
# # -------------------------------------------------

edges_list = [
    ('BOM', 'SOL'),
    ('BEL', 'HUB'),
    ('BOM', 'HUB'),
    ('SOL', 'GUL'),
    ('HUB', 'GUL'),
    ('HUB', 'BAN'),
    ('GUL', 'BAN'),
    ('GUL', 'CHE'),
    ('BAN', 'CHE')
]

supply_dict = [{'BOM': 220}, {'BEL': 130}]
demand_dict = [{'CHE': 200}, {'BAN': 100}]

cost = [3, 2, 6, 4, 4, 4, 2, 7, 3]

# Convert lists of dictionaries into dictionaries
supply = {}
for item in supply_dict:
    supply.update(item)

demand = {}
for item in demand_dict:
    demand.update(item)

cost_dict = {
    edges_list[k]: cost[k]
    for k in range(len(edges_list))
}

# # -------------------------------------------------
# # 2. Supply-demand imbalance
# # -------------------------------------------------

total_supply = sum(supply.values())
total_demand = sum(demand.values())

print("Total supply:", total_supply)
print("Total demand:", total_demand)

if total_supply > total_demand:
    surplus = total_supply - total_demand
    print("Surplus supply:", surplus)

    # Dummy node absorbs the unused supply
    dummy_node = 'DUMMY'

    demand[dummy_node] = surplus

    # Add zero-cost arcs from supply nodes to dummy node
    for node in supply:
        edges_list.append((node, dummy_node))
        cost_dict[(node, dummy_node)] = 0

elif total_demand > total_supply:
    deficit = total_demand - total_supply
    raise ValueError(
        "Demand exceeds supply by {}. "
        "Add a dummy supply node or revise the data.".format(deficit)
    )

# # -------------------------------------------------
# # 3. Construct node balances
# # -------------------------------------------------

nodes = set()
for i, j in edges_list:
    nodes.add(i)
    nodes.add(j)

balance = {n: 0 for n in nodes}

for n, quantity in supply.items():
    balance[n] += quantity

for n, quantity in demand.items():
    balance[n] -= quantity

# # -------------------------------------------------
# # 4. Build the optimization model
# # -------------------------------------------------

model = pyo.ConcreteModel()

model.N = pyo.Set(initialize=sorted(nodes))
model.A = pyo.Set(dimen=2, initialize=edges_list)

model.x = pyo.Var(model.A, domain=pyo.NonNegativeReals)

# -------------------------------------------------
# 5. Objective: minimize transportation cost
# -------------------------------------------------

model.obj = pyo.Objective(
    expr=sum(
        cost_dict[i, j] * model.x[i, j]
        for i, j in edges_list
    ),
    sense=pyo.minimize
)

# -------------------------------------------------
# 6. Flow balance constraints
# -------------------------------------------------

def flow_balance_rule(model, n):
    outgoing = sum(
        model.x[i, j]
        for i, j in edges_list if i == n
    )

    incoming = sum(
        model.x[i, j]
        for i, j in edges_list if j == n
    )

    return outgoing - incoming == balance[n]

model.flow_balance = pyo.Constraint(
    model.N,
    rule=flow_balance_rule
)

# -------------------------------------------------
# 7. Solve the model
# -------------------------------------------------

solver = pyo.SolverFactory('scip')

if not solver.available(exception_flag=False):
    raise RuntimeError(
        "Install SCIP using: pip install pyscipopt"
    )

results = solver.solve(model, tee=False)

# -------------------------------------------------
# 8. Display results
# -------------------------------------------------

if results.solver.termination_condition == \
        pyo.TerminationCondition.optimal:

    print("\nOptimal transportation cost:",
          pyo.value(model.obj))

    print("\nOptimal flow on each edge:")

    for i, j in edges_list:
        flow = pyo.value(model.x[i, j])
        if flow > 1e-6:
            print("{} -> {}: {:.2f}".format(i, j, flow))

    if total_supply > total_demand:
        unused = sum(
            pyo.value(model.x[node, 'DUMMY'])
            for node in supply
        )
        print("\nUnused supply:", round(unused, 2))

else:
    print("Solver status:", results.solver.status)
    print("Termination condition:",
          results.solver.termination_condition)
