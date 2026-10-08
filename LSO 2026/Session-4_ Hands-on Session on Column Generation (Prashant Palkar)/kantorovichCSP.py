# # ============================================================
# # Cutting Stock Problem - Kantorovich Formulation
# #
# # Data files:
# #
# # stock<instance Id>.csv
# # items<instance Id>.csv

# # ============================================================
# import time

# from dataclasses import dataclass

# import pandas as pd
# import gurobipy as gp
# from gurobipy import GRB


# # ============================================================
# # Problem instance
# # ============================================================

# @dataclass
# class CuttingStockInstance:
#     stock_length: int
#     items: list
#     lengths: dict
#     demands: dict

#     @property
#     def max_rolls(self):
#         """
#         Simple upper bound:
#         one roll per demanded item.
#         """
#         return sum(self.demands.values())

#     @classmethod
#     def from_csv(
#         cls,
#         stock_file="stock1.csv",
#         items_file="items1.csv"
#     ):

#         stock_df = pd.read_csv(stock_file)
#         items_df = pd.read_csv(items_file)

#         stock_length = int(
#             stock_df.loc[0, "stock_length"]
#         )

#         items = items_df["item"].tolist()

#         lengths = dict(
#             zip(
#                 items_df["item"],
#                 items_df["length"]
#             )
#         )

#         demands = dict(
#             zip(
#                 items_df["item"],
#                 items_df["demand"]
#             )
#         )

#         return cls(
#             stock_length=stock_length,
#             items=items,
#             lengths=lengths,
#             demands=demands
#         )


# # ============================================================
# # Kantorovich Model
# #
# # Variables
# # ----------
# # x[i,j] = number of items of type i cut from roll j
# # y[j]   = 1 if roll j is used
# #
# # Objective
# # ----------
# # Minimize number of rolls used
# #
# # Constraints
# # ----------
# # Demand satisfaction
# # Capacity constraints
# # Linking constraints
# # Symmetry breaking
# # ============================================================

# class CuttingStockKantorovich:

#     def __init__(self, instance):

#         self.data = instance

#         self.model = gp.Model(
#             "CuttingStock_Kantorovich"
#         )

#         self.x = None
#         self.y = None

#     # --------------------------------------------------------
#     # Build model
#     # --------------------------------------------------------

#     def build(self):

#         I = self.data.items
#         J = range(self.data.max_rolls)

#         # Decision variables

#         self.x = self.model.addVars(
#             I,
#             J,
#             vtype=GRB.INTEGER,
# 			# If relaxing x only, comment the above line and uncomment the following line
#             # vtype=GRB.CONTINUOUS,
#             lb=0,
#             name="x"
#         )

#         self.y = self.model.addVars(
#             J,
#             vtype=GRB.BINARY,
# 			# If relaxing y only, comment the above line and uncomment the following three lines
#             # vtype=GRB.CONTINUOUS,
#             # lb=0,
#             # ub=1,
#             name="y"
#         )

#         # Objective:
#         # minimize number of rolls used

#         self.model.setObjective(
#             gp.quicksum(
#                 self.y[j]
#                 for j in J
#             ),
#             GRB.MINIMIZE
#         )

#         # ----------------------------------------------------
#         # Demand constraints
#         #
#         # Produce at least demand[i]
#         # ----------------------------------------------------

#         for i in I:

#             self.model.addConstr(
#                 gp.quicksum(
#                     self.x[i, j]
#                     for j in J
#                 )
#                 >= self.data.demands[i],
#                 name=f"demand_{i}"
#             )

#         # ----------------------------------------------------
#         # Capacity constraints
#         #
#         # Total cut length on roll j
#         # cannot exceed stock length
#         # ----------------------------------------------------

#         for j in J:

#             self.model.addConstr(
#                 gp.quicksum(
#                     self.data.lengths[i]
#                     * self.x[i, j]
#                     for i in I
#                 )
#                 <= self.data.stock_length
#                 * self.y[j],
#                 name=f"capacity_{j}"
#             )

#         # ----------------------------------------------------
#         # Symmetry-breaking constraints
#         #
#         # y1 >= y2 >= y3 >= ...
#         # ----------------------------------------------------

#         # for j in range(self.data.max_rolls - 1):

#             # self.model.addConstr(
#                 # self.y[j]
#                 # >= self.y[j + 1],
#                 # name=f"symmetry_{j}"
#             # )


#     # --------------------------------------------------------
#     # Solve model
#     # --------------------------------------------------------

#     def solve(self):

#         self.model.optimize()

#     # --------------------------------------------------------
#     # Print solution as pattern table
#     # --------------------------------------------------------

#     def print_solution(self):

#         if self.model.Status != GRB.OPTIMAL:
#             print("No optimal solution found.")
#             return

#         I = self.data.items
#         J = range(self.data.max_rolls)

#         print("\n")
#         print("=" * 80)
#         print(
#             f"Optimal rolls used = "
#             f"{int(round(self.model.ObjVal))}"
#         )
#         print("=" * 80)

#         # Header uses item lengths

#         header = ["Roll"]

#         for i in I:
#             header.append(
#                 str(self.data.lengths[i])
#             )

#         header.extend(
#             ["Used", "Waste"]
#         )

#         print(
#             "".join(
#                 f"{h:>10}"
#                 for h in header
#             )
#         )

#         print(
#             "-" * (10 * len(header))
#         )

#         roll_count = 0

#         for j in J:

#             if self.y[j].X > 0.5:

#                 roll_count += 1

#                 pattern = []

#                 for i in I:

#                     pattern.append(
#                         int(
#                             round(
#                                 self.x[i, j].X
#                             )
#                         )
#                     )

#                 used = sum(
#                     self.data.lengths[i]
#                     * pattern[k]
#                     for k, i in enumerate(I)
#                 )

#                 # waste = (
#                     # self.data.stock_length
#                     # - used
#                 # )
#                 waste = self.data.stock_length - used
#                 # If relaxing y variables
#                 waste = self.data.stock_length * self.y[j].X - used

#                 row = [
#                     str(roll_count)
#                 ]

#                 row.extend(
#                     str(v)
#                     for v in pattern
#                 )

#                 row.append(str(used))
#                 row.append(str(waste))

#                 print(
#                     "".join(
#                         f"{v:>10}"
#                         for v in row
#                     )
#                 )

# # ============================================================
# # Main
# # ============================================================

# def main():

#     # -----------------------------
#     # Start wall-clock timer
#     # -----------------------------
#     start_time = time.perf_counter()

#     # Read instance

#     instance = (
#         CuttingStockInstance
#         .from_csv(
#             "stock1.csv",
#             "items1.csv"
#         )
#     )

#     # Build model

#     cs = CuttingStockKantorovich(
#         instance
#     )

#     cs.build()

#     # Optional solver settings

#     cs.model.Params.OutputFlag = 1

#     # LP relaxation (optional)
#     # lp = cs.model.relax()
#     # lp.optimize()
#     # print("LP Status:", lp.Status)
#     # print("LP objective:", lp.ObjVal)

#     # =================================
#     # LP relaxation
#     # =================================
#     ## Relax y variables
#     # for j in range(cs.data.max_rolls):
#         # cs.y[j].VType = GRB.CONTINUOUS
#         # cs.y[j].LB = 0.0
#         # cs.y[j].UB = 1.0
    
#     # # # Relax x variables
#     # for i in cs.data.items:
#         # for j in range(cs.data.max_rolls):
#             # cs.x[i, j].VType = GRB.CONTINUOUS
#             # cs.x[i, j].LB = 0.0
    
#     # # Solve LP relaxation
#     # cs.model.optimize()
    
#     # # Print LP results
#     # print("LP Status:", cs.model.Status)
#     # print("LP objective:", cs.model.ObjVal)
#     # =================================

#     # Solve MIP
#     cs.model.optimize()

#     # Solve
#     # cs.solve()

#     # Print solution
#     # print("\nSolution summary:\n")
#     # print("\nX variables (non-zero):")
#     # for (i, j), var in cs.x.items():
#         # if var.X > 1e-6:
#             # print(f"x[{i},{j}] = {var.X}")
    
#     print("\nY variables (used rolls):")
#     for j, var in cs.y.items():
#         if var.X > 0.5:
#             print(f"y[{j}] = {var.X}")    
#     # cs.print_solution()

#     # -----------------------------
#     # Wall-clock time (total)
#     # -----------------------------
#     total_time = time.perf_counter() - start_time

#     # -----------------------------
#     # Solver time (Gurobi only)
#     # -----------------------------
#     solve_time = cs.model.Runtime

#     # -----------------------------
#     # Print timing summary
#     # -----------------------------
#     print("\n" + "=" * 80)
#     print(f"Total wall-clock time : {total_time:.4f} s")
#     print(f"Gurobi solve time     : {solve_time:.4f} s")
#     print("=" * 80)

# if __name__ == "__main__":
#     main()



# ============================================================
# Cutting Stock Problem - Kantorovich Formulation
#
# Data files:
#
# stock<instance Id>.csv
# items<instance Id>.csv
# ============================================================
import time
import os
from dataclasses import dataclass

import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# ============================================================
# Problem instance
# ============================================================

@dataclass
class CuttingStockInstance:
    stock_length: int
    items: list
    lengths: dict
    demands: dict

    @property
    def max_rolls(self):
        """
        Simple upper bound:
        one roll per demanded item.
        """
        return sum(self.demands.values())

    @classmethod
    def from_csv(cls, stock_file="stock1.csv", items_file="items1.csv"):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stock_path = os.path.join(script_dir, stock_file)
        items_path = os.path.join(script_dir, items_file)

        # If files do not exist, create default ones
        if not os.path.exists(stock_path):
            print(f"File {stock_file} not found. Creating a default one.")
            pd.DataFrame({"stock_length": [100]}).to_csv(stock_path, index=False)

        if not os.path.exists(items_path):
            print(f"File {items_file} not found. Creating a default one.")
            default_items = pd.DataFrame({
                "item": ["A", "B", "C"],
                "length": [25, 40, 30],
                "demand": [8, 5, 6]
            })
            default_items.to_csv(items_path, index=False)

        stock_df = pd.read_csv(stock_path)
        items_df = pd.read_csv(items_path)

        stock_length = int(stock_df.loc[0, "stock_length"])
        items = items_df["item"].tolist()
        lengths = dict(zip(items_df["item"], items_df["length"]))
        demands = dict(zip(items_df["item"], items_df["demand"]))

        return cls(
            stock_length=stock_length,
            items=items,
            lengths=lengths,
            demands=demands
        )


# ============================================================
# Kantorovich Model
# ============================================================

class CuttingStockKantorovich:

    def __init__(self, instance):
        self.data = instance
        self.model = gp.Model("CuttingStock_Kantorovich")
        self.x = None
        self.y = None

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    def build(self):
        I = self.data.items
        J = range(self.data.max_rolls)

        # Decision variables
        self.x = self.model.addVars(
            I, J, vtype=GRB.INTEGER, lb=0, name="x"
        )
        self.y = self.model.addVars(
            J, vtype=GRB.BINARY, name="y"
        )

        # Objective: minimize number of rolls used
        self.model.setObjective(
            gp.quicksum(self.y[j] for j in J),
            GRB.MINIMIZE
        )

        # Demand constraints
        for i in I:
            self.model.addConstr(
                gp.quicksum(self.x[i, j] for j in J) >= self.data.demands[i],
                name=f"demand_{i}"
            )

        # Capacity constraints
        for j in J:
            self.model.addConstr(
                gp.quicksum(self.data.lengths[i] * self.x[i, j] for i in I)
                <= self.data.stock_length * self.y[j],
                name=f"capacity_{j}"
            )

        # Optional symmetry-breaking constraints (commented by default)
        # for j in range(self.data.max_rolls - 1):
        #     self.model.addConstr(self.y[j] >= self.y[j+1], name=f"symmetry_{j}")

    # --------------------------------------------------------
    # Solve model
    # --------------------------------------------------------
    def solve(self):
        self.model.optimize()

    # --------------------------------------------------------
    # Print solution as pattern table
    # --------------------------------------------------------
    def print_solution(self):
        if self.model.Status != GRB.OPTIMAL:
            print("No optimal solution found.")
            return

        I = self.data.items
        J = range(self.data.max_rolls)

        print("\n")
        print("=" * 80)
        print(f"Optimal rolls used = {int(round(self.model.ObjVal))}")
        print("=" * 80)

        header = ["Roll"] + [str(self.data.lengths[i]) for i in I] + ["Used", "Waste"]
        print("".join(f"{h:>10}" for h in header))
        print("-" * (10 * len(header)))

        roll_count = 0
        for j in J:
            if self.y[j].X > 0.5:
                roll_count += 1
                pattern = [int(round(self.x[i, j].X)) for i in I]
                used = sum(self.data.lengths[i] * pattern[k] for k, i in enumerate(I))
                waste = self.data.stock_length - used  # if y[j] is binary
                # If y is relaxed, use: waste = self.data.stock_length * self.y[j].X - used

                row = [str(roll_count)] + [str(v) for v in pattern] + [str(used), str(waste)]
                print("".join(f"{v:>10}" for v in row))


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.perf_counter()

    # Read instance (files are now located automatically)
    instance = CuttingStockInstance.from_csv("stock1.csv", "items1.csv")

    cs = CuttingStockKantorovich(instance)
    cs.build()

    # Optional solver settings
    cs.model.Params.OutputFlag = 1

    # Solve MIP
    cs.model.optimize()

    # Print used rolls
    print("\nY variables (used rolls):")
    for j, var in cs.y.items():
        if var.X > 0.5:
            print(f"y[{j}] = {var.X}")

    cs.print_solution()

    total_time = time.perf_counter() - start_time
    solve_time = cs.model.Runtime

    print("\n" + "=" * 80)
    print(f"Total wall-clock time : {total_time:.4f} s")
    print(f"Gurobi solve time     : {solve_time:.4f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()