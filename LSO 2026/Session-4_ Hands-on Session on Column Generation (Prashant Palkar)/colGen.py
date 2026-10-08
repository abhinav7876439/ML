import time
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass


# ============================================================
# Problem Instance
# ============================================================

@dataclass
class CuttingStockInstance:
    stock_length: int
    items: list
    lengths: dict
    demands: dict

    @classmethod
    def from_csv(cls, stock_file="stock3.csv", items_file="items3.csv"):

        stock_df = pd.read_csv(stock_file)
        items_df = pd.read_csv(items_file)

        stock_length = int(stock_df.loc[0, "stock_length"])

        items = items_df["item"].tolist()
        lengths = dict(zip(items_df["item"], items_df["length"]))
        demands = dict(zip(items_df["item"], items_df["demand"]))

        return cls(stock_length, items, lengths, demands)


# ============================================================
# Column Generation Solver
# ============================================================

class CuttingStockDCG:

    def __init__(self, instance):
        self.data = instance

        self.master = gp.Model("CuttingStock_RMP")
        self.master.Params.OutputFlag = 1

        self.patterns = []   # list of dict patterns
        self.x = []          # master variables (one per pattern)

        self.demand_constr = {}

    # --------------------------------------------------------
    # Initial patterns (1-item greedy patterns)
    # --------------------------------------------------------
    def initialize_patterns(self):

        I = self.data.items
        L = self.data.stock_length

        for i in I:
            pattern = {j: 0 for j in I}
            pattern[i] = L // self.data.lengths[i]
            self.patterns.append(pattern)

    # --------------------------------------------------------
    # Randomized Greedy Patterns 
    # --------------------------------------------------------

    # --------------------------------------------------------
    # First-Fit Decreasing Patterns
    # --------------------------------------------------------
    def ffd_initialize_patterns(self, instance):
    
        L = instance.stock_length
    
        # sort items by decreasing size
        items_sorted = sorted(
            instance.items,
            key=lambda i: instance.lengths[i],
            reverse=True
        )
    
        # remaining demand copy (important!)
        remaining_demand = instance.demands.copy()
    
        patterns = []
    
        # while there is still demand left
        while sum(remaining_demand.values()) > 0:
    
            remaining_capacity = L
            pattern = {i: 0 for i in instance.items}
    
            # fill one roll greedily
            for i in items_sorted:
    
                if remaining_demand[i] <= 0:
                    continue
    
                max_fit = min(
                    remaining_demand[i],
                    remaining_capacity // instance.lengths[i]
                )
    
                if max_fit > 0:
                    pattern[i] = max_fit
                    remaining_demand[i] -= max_fit
                    remaining_capacity -= max_fit * instance.lengths[i]
    
            patterns.append(pattern)
    
        return patterns
    # --------------------------------------------------------
    # Build Restricted Master Problem (RMP)
    # --------------------------------------------------------
    def build_master(self):

        I = self.data.items

        for p, pattern in enumerate(self.patterns):

            col = gp.Column()

            for i in I:
                if pattern[i] > 0:
                    col.addTerms(pattern[i], self.demand_constr.get(i))

            var = self.master.addVar(
                lb=0,
                vtype=GRB.CONTINUOUS,
                obj=1,
                column=col,
                name=f"x_{p}"
            )

            self.x.append(var)

        # Demand constraints
        for i in I:
            self.demand_constr[i] = self.master.addConstr(
                gp.LinExpr() >= self.data.demands[i],
                name=f"demand_{i}"
            )

    # --------------------------------------------------------
    # Solve master problem
    # --------------------------------------------------------
    def solve_master(self):
        self.master.optimize()

    # --------------------------------------------------------
    # Pricing problem (Knapsack)
    # --------------------------------------------------------
    def pricing(self, duals):

        I = self.data.items
        L = self.data.stock_length

        pricing = gp.Model("pricing")
        pricing.Params.OutputFlag = 0

        y = pricing.addVars(I, vtype=GRB.INTEGER, lb=0, name="y")

        pricing.setObjective(
            gp.quicksum(duals[i] * y[i] for i in I),
            GRB.MAXIMIZE
        )

        pricing.addConstr(
            gp.quicksum(self.data.lengths[i] * y[i] for i in I)
            <= L
        )

        pricing.optimize()

        if pricing.Status != GRB.OPTIMAL:
            return None, 0

        reduced_cost = 1 - pricing.ObjVal

        if reduced_cost < -1e-6:
            pattern = {i: int(y[i].X) for i in I}
            return pattern, reduced_cost

        return None, reduced_cost

    # --------------------------------------------------------
    # Column Generation Loop
    # --------------------------------------------------------
    def solve(self):

        # Initialize patterns
        # self.initialize_patterns()
        self.patterns = self.ffd_initialize_patterns(self.data)

        # IMPORTANT: build constraints first (needed for columns)
        I = self.data.items
        for i in I:
            self.demand_constr[i] = self.master.addConstr(
                gp.LinExpr() >= self.data.demands[i],
                name=f"demand_{i}"
            )

        # now add initial columns
        for p, pattern in enumerate(self.patterns):

            col = gp.Column()
            for i in I:
                col.addTerms(pattern[i], self.demand_constr[i])

            var = self.master.addVar(
                lb=0,
                vtype=GRB.CONTINUOUS,
                obj=1,
                column=col,
                name=f"x_{p}"
            )

            self.x.append(var)

        self.master.update()

        iteration = 0

        while True:

            iteration += 1
            print(f"\nIteration {iteration}")

            self.solve_master()

            if self.master.Status != GRB.OPTIMAL:
                print("Master not optimal")
                break

            duals = {i: self.demand_constr[i].Pi for i in I}

            pattern, rc = self.pricing(duals)

            print("Reduced cost:", rc)

            if pattern is None:
                print("Optimal LP reached (no improving column).")
                break

            print("Adding pattern:", pattern)

            self.patterns.append(pattern)

            # build column
            col = gp.Column()
            for i in I:
                col.addTerms(pattern[i], self.demand_constr[i])

            var = self.master.addVar(
                lb=0,
                vtype=GRB.CONTINUOUS,
                obj=1,
                column=col,
                name=f"x_{len(self.x)}"
            )

            self.x.append(var)

            self.master.update()

        print("\nFinal LP objective:", self.master.ObjVal)


# ============================================================
# Main
# ============================================================

def main():

    start = time.perf_counter()

    instance = CuttingStockInstance.from_csv(
        "stock3.csv",
        "items3.csv"
    )

    solver = CuttingStockDCG(instance)
    solver.solve()

    print("\nTotal time:", round(time.perf_counter() - start, 4), "s")


if __name__ == "__main__":
    main()




# import time
# import os
# import pandas as pd
# import gurobipy as gp
# from gurobipy import GRB
# from dataclasses import dataclass

# # ============================================================
# # Problem Instance
# # ============================================================

# @dataclass
# class CuttingStockInstance:
#     stock_length: int
#     items: list
#     lengths: dict
#     demands: dict

#     @classmethod
#     def from_csv(cls, stock_file="stock3.csv", items_file="items3.csv"):
#         # If files do not exist, create them with sample data
#         if not os.path.exists(stock_file):
#             pd.DataFrame({"stock_length": [100]}).to_csv(stock_file, index=False)
#             print(f"Created sample {stock_file}")
#         if not os.path.exists(items_file):
#             sample_items = pd.DataFrame({
#                 "item": ["A", "B", "C"],
#                 "length": [25, 40, 30],
#                 "demand": [8, 5, 6]
#             })
#             sample_items.to_csv(items_file, index=False)
#             print(f"Created sample {items_file}")

#         stock_df = pd.read_csv(stock_file)
#         items_df = pd.read_csv(items_file)

#         stock_length = int(stock_df.loc[0, "stock_length"])
#         items = items_df["item"].tolist()
#         lengths = dict(zip(items_df["item"], items_df["length"]))
#         demands = dict(zip(items_df["item"], items_df["demand"]))

#         return cls(stock_length, items, lengths, demands)


# # ============================================================
# # Column Generation Solver
# # ============================================================

# class CuttingStockDCG:

#     def __init__(self, instance):
#         self.data = instance
#         self.master = gp.Model("CuttingStock_RMP")
#         self.master.Params.OutputFlag = 1   # set to 0 to suppress Gurobi output

#         self.patterns = []   # list of patterns (dict item -> count)
#         self.x = []          # master variables (one per pattern)
#         self.demand_constr = {}

#     # --------------------------------------------------------
#     # First-Fit Decreasing Patterns (robust version)
#     # --------------------------------------------------------
#     def ffd_initialize_patterns(self, instance):
#         L = instance.stock_length

#         # sort items by decreasing length
#         items_sorted = sorted(
#             instance.items,
#             key=lambda i: instance.lengths[i],
#             reverse=True
#         )

#         remaining_demand = instance.demands.copy()
#         patterns = []

#         # safety: avoid infinite loop if some item is longer than stock
#         max_item_len = max(instance.lengths.values())
#         if max_item_len > L:
#             raise ValueError(f"Item length {max_item_len} exceeds stock length {L}")

#         while sum(remaining_demand.values()) > 0:
#             remaining_capacity = L
#             pattern = {i: 0 for i in instance.items}

#             # fill one roll greedily
#             for i in items_sorted:
#                 if remaining_demand[i] <= 0:
#                     continue

#                 max_fit = min(
#                     remaining_demand[i],
#                     remaining_capacity // instance.lengths[i]
#                 )

#                 if max_fit > 0:
#                     pattern[i] = max_fit
#                     remaining_demand[i] -= max_fit
#                     remaining_capacity -= max_fit * instance.lengths[i]

#             # if no item could be placed, break (should not happen due to earlier check)
#             if all(v == 0 for v in pattern.values()):
#                 break

#             patterns.append(pattern)

#         return patterns

#     # --------------------------------------------------------
#     # Pricing problem (knapsack)
#     # --------------------------------------------------------
#     def pricing(self, duals):
#         I = self.data.items
#         L = self.data.stock_length

#         pricing = gp.Model("pricing")
#         pricing.Params.OutputFlag = 0

#         y = pricing.addVars(I, vtype=GRB.INTEGER, lb=0, name="y")

#         # maximize sum(dual_i * y_i)
#         pricing.setObjective(
#             gp.quicksum(duals[i] * y[i] for i in I),
#             GRB.MAXIMIZE
#         )

#         pricing.addConstr(
#             gp.quicksum(self.data.lengths[i] * y[i] for i in I) <= L
#         )

#         pricing.optimize()

#         if pricing.Status != GRB.OPTIMAL:
#             return None, 0

#         reduced_cost = 1 - pricing.ObjVal

#         if reduced_cost < -1e-6:
#             pattern = {i: int(y[i].X) for i in I}
#             return pattern, reduced_cost

#         return None, reduced_cost

#     # --------------------------------------------------------
#     # Column Generation Loop
#     # --------------------------------------------------------
#     def solve(self):
#         # 1. Generate initial patterns (FFD)
#         self.patterns = self.ffd_initialize_patterns(self.data)
#         print(f"Initialized with {len(self.patterns)} patterns")

#         # 2. Build demand constraints (empty RHS first)
#         I = self.data.items
#         for i in I:
#             self.demand_constr[i] = self.master.addConstr(
#                 gp.LinExpr() >= self.data.demands[i],
#                 name=f"demand_{i}"
#             )

#         # 3. Add initial columns
#         for p, pattern in enumerate(self.patterns):
#             col = gp.Column()
#             for i in I:
#                 if pattern[i] > 0:
#                     col.addTerms(pattern[i], self.demand_constr[i])

#             var = self.master.addVar(
#                 lb=0, vtype=GRB.CONTINUOUS, obj=1,
#                 column=col, name=f"x_{p}"
#             )
#             self.x.append(var)

#         self.master.update()

#         iteration = 0
#         while True:
#             iteration += 1
#             print(f"\n--- Column Generation Iteration {iteration} ---")

#             # Solve restricted master
#             self.master.optimize()
#             if self.master.Status != GRB.OPTIMAL:
#                 print("Master problem is not optimal (maybe infeasible?).")
#                 break

#             # Retrieve duals (for >= constraints, duals are >= 0)
#             duals = {i: self.demand_constr[i].Pi for i in I}

#             # Solve pricing problem
#             pattern, rc = self.pricing(duals)
#             print(f"Reduced cost: {rc:.6f}")

#             if pattern is None or rc >= -1e-6:
#                 print("Optimal LP reached (no improving column).")
#                 break

#             print("Adding new pattern:", pattern)
#             self.patterns.append(pattern)

#             # Add the new column
#             col = gp.Column()
#             for i in I:
#                 if pattern[i] > 0:
#                     col.addTerms(pattern[i], self.demand_constr[i])

#             var = self.master.addVar(
#                 lb=0, vtype=GRB.CONTINUOUS, obj=1,
#                 column=col, name=f"x_{len(self.x)}"
#             )
#             self.x.append(var)
#             self.master.update()

#         print(f"\nFinal LP objective (lower bound): {self.master.ObjVal:.4f}")

#         # ----------------------------------------------------
#         # Optional: Round the LP solution to an integer solution
#         # ----------------------------------------------------
#         self._round_solution()

#     def _round_solution(self):
#         """Simple rounding heuristic: solve the master as an IP using only generated patterns."""
#         print("\n--- Solving Integer Master (Rounding Heuristic) ---")
#         int_model = gp.Model("CuttingStock_IP")
#         int_model.Params.OutputFlag = 0

#         # Integer variables
#         x_int = int_model.addVars(len(self.patterns), vtype=GRB.INTEGER, lb=0, name="x")
#         int_model.setObjective(gp.quicksum(x_int), GRB.MINIMIZE)

#         # Demand constraints
#         for i in self.data.items:
#             int_model.addConstr(
#                 gp.quicksum(self.patterns[p][i] * x_int[p] for p in range(len(self.patterns))) >= self.data.demands[i],
#                 name=f"demand_{i}"
#             )

#         int_model.optimize()
#         if int_model.Status == GRB.OPTIMAL:
#             print(f"Integer solution uses {int_model.ObjVal:.0f} rolls.")
#             # Display pattern usage
#             for p, var in x_int.items():
#                 if var.X > 0.5:
#                     print(f"  Pattern {p}: {self.patterns[p]} used {int(var.X)} times")
#         else:
#             print("Could not find integer solution.")


# # ============================================================
# # Main
# # ============================================================

# def main():
#     start = time.perf_counter()

#     instance = CuttingStockInstance.from_csv("stock3.csv", "items3.csv")
#     print(f"Stock length: {instance.stock_length}")
#     print("Items:", list(instance.items))
#     print("Lengths:", instance.lengths)
#     print("Demands:", instance.demands)

#     solver = CuttingStockDCG(instance)
#     solver.solve()

#     print(f"\nTotal time: {time.perf_counter() - start:.4f} s")


# if __name__ == "__main__":
#     main()