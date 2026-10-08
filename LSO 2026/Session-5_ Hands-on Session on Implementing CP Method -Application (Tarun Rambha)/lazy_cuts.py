import gurobipy as gp
from gurobipy import GRB


def lazy_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        x_val = model.cbGetSolution(model._vars)  # get current integer solution

        # Check if the lazy constraint x[0] + x[1] <= 1 is violated
        if x_val[0] + x_val[1] > 1 + 1e-6:
            print("Lazy constraint violated: x[0] + x[1] <= 1, adding cut.")
            print(f"Current solution: "
                  f"x[0]={x_val[0]}, "
                  f"x[1]={x_val[1]}, "
                  f"x[2]={x_val[2]}")
            # Add the violated constraint as a lazy cut
            model.cbLazy(model._vars[0] + model._vars[1] <= 1)

m = gp.Model()
m.Params.LazyConstraints = 1  # enable a flag

x = m.addVars(3, vtype=GRB.BINARY, obj=1, name="x")  # maximize x[0]+x[1]+x[2]
m._vars = x  # allows callback to access variables

m.ModelSense = GRB.MAXIMIZE
m.addConstr(x[0] + x[1] + x[2] <= 2, "budget")
# x[0] + x[1] <= 1 is intentionally NOT added upfront

m.optimize(lazy_callback)

print([x[i].X for i in range(3)])
