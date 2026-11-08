from pyomo.environ import *  



SOLVER = "cplex"  # "glpk" or "cbc" .

edges = [  # Specify the arcs (from node, to node).
    ('BOM', 'SOL'), ('BEL', 'HUB'), ('BOM', 'HUB'),
    ('SOL', 'GUL'), ('HUB', 'GUL'), ('HUB', 'BAN'),
    ('GUL', 'BAN'), ('GUL', 'CHE'), ('BAN', 'CHE')
]
cost = [3, 2, 6, 4, 4, 4, 2, 7, 3]  # cost per million barrels on each arc (same order as edges).

supply = {'BOM': 190, 'BEL': 110}  # supply available at each source node.

scenarios = {  # demand at each demand node for every scenario.
    "Balanced   (supply = demand)": {'CHE': 200, 'BAN': 100},
    "Unbalanced (supply > demand)": {'CHE': 70, 'BAN': 40},
}

# Optional disruptions (set to None to switch off).
# BLOCKED_ARCS = []  # Example: [('SOL', 'GUL')] forces zero flow on these arcs.
# NODE_INFLOW_CAP = {}  # Example: {'BAN': 150} limits total inflow into a node.

BLOCKED_ARCS = [('SOL', 'GUL')]
NODE_INFLOW_CAP = {'BAN': 150}

nodes = sorted({n for arc in edges for n in arc})  # Collect all unique nodes from the arcs.
arc_cost = dict(zip(edges, cost))  # Map each arc to its transportation cost.




def solve_network(demand):
    total_supply = sum(supply.values())  # Compute the total supply.
    total_demand = sum(demand.values())  # Compute the total demand.

    
    model = ConcreteModel()  # Initialize the Pyomo optimization model.
    model.x = Var(edges, domain=NonNegativeReals)  # x[i,j] = flow sent on arc (i,j).

    model.balance = ConstraintList()  

    for n in nodes:  
        outflow = sum(model.x[i, j] for (i, j) in edges if i == n)  # Flow leaving node n.
        inflow = sum(model.x[i, j] for (i, j) in edges if j == n)  # Flow entering node n.

        if n in supply:  # Source node: may ship out up to its supply.
            model.balance.add(outflow - inflow <= supply[n])
        elif n in demand:  # Demand node: net inflow must equal its demand exactly.
            model.balance.add(inflow - outflow == demand[n])
        else:  # Transshipment node: whatever enters must leave.
            model.balance.add(inflow == outflow)

    
    for (i, j) in BLOCKED_ARCS:  # Close disrupted arcs.
        model.x[i, j].fix(0)

    for n, cap in NODE_INFLOW_CAP.items():  # Limit inflow at capacity-restricted nodes.
        model.balance.add(sum(model.x[i, j] for (i, j) in edges if j == n) <= cap)

    # MINIMIZE TOTAL TRANSPORTATION COST
    model.objective = Objective(  
        expr=sum(arc_cost[a] * model.x[a] for a in edges),  # Sum cost x flow over all arcs.
        sense=minimize  # Minimize total transportation cost.
    )

    
    solver = SolverFactory(SOLVER)  
    results = solver.solve(model, tee=False)  

    return model, results, total_supply, total_demand


# RUN AND DISPLAY EACH SCENARIO

for name, demand in scenarios.items():  # Loop over the balanced and unbalanced scenarios.
    model, results, total_supply, total_demand = solve_network(demand)

    print("\n" + "=" * 55) 
    print("Scenario:", name)  
    print("Total supply =", total_supply, "| Total demand =", total_demand)

    if results.solver.termination_condition != TerminationCondition.optimal:  # Check the solution status.
        print("No optimal solution. Status:", results.solver.termination_condition)
        continue

    print("\nFlow on each arc:")  
    for (i, j) in edges:  
        flow = value(model.x[i, j])  
        if flow > 1e-6:  
            print(f"  {i} -> {j}: {flow:.2f}")

    print("\nOil shipped out of each source:")  
    for s in supply:  
        used = sum(value(model.x[i, j]) for (i, j) in edges if i == s)  
        print(f"  {s}: {used:.2f} used of {supply[s]} available")

    print("\nMinimum total cost =", round(value(model.objective), 2))  