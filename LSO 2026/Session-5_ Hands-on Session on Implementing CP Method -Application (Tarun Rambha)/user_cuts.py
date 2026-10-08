import gurobipy as gp
from gurobipy import GRB


# Knapsack: maximize profit subject to weight <= capacity
# Items: (weight, profit) = (3, 4), (4, 5), (5, 6)
# Capacity = 7
# Optimal integer solution: pick items 0 and 1 → weight=7, profit=9

def user_cut_callback(model, where):
    if where == GRB.Callback.MIPNODE:
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x_val = model.cbGetNodeRel(model._vars)

            # Cover inequality: items {1, 2} form a cover (weight 4+5=9 > 7)
            # So x[1] + x[2] <= 1 is valid for ALL integer solutions
            # (you can never pick both items 1 and 2 — they exceed capacity)
            # But the LP relaxation might have x[1]=0.8, x[2]=0.6 → sum=1.4 > 1
            if x_val[1] + x_val[2] > 1 + 1e-6:
                print(
                    f"  Adding cover cut: x[1]+x[2] <= 1  (LP vals: {x_val[1]:.2f}, {x_val[2]:.2f})")
                model.cbCut(model._vars[1] + model._vars[2] <= 1)


m = gp.Model()
m.Params.OutputFlag = 0

weights = [3, 4, 5]
profits = [4, 5, 6]
capacity = 7

x = m.addVars(3, vtype=GRB.BINARY, name="x")
m._vars = x

m.setObjective(gp.quicksum(profits[i] * x[i] for i in range(3)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum(weights[i] * x[i] for i in range(3)) <= capacity,
            "capacity")

m.optimize(user_cut_callback)

print(f"\nSolution: {[int(x[i].X) for i in range(3)]}")
print(f"x[1] + x[2] = {int(x[1].X) + int(x[2].X)} <= 1")  # guaranteed now
