from pyomo.environ import *
import pandas as pd
import numpy as np
import re
import time



# Read the Excel file
# df = pd.read_csv('transportation_50x30_balanced.csv')
# df_fixed = pd.read_csv('transportation_50x30_fixed_costs.csv')



df = pd.read_csv( r"C:\Users\abhin\Desktop\Abhi\AOTM 2026\transportation_50x30_balanced.csv")

df_fixed = pd.read_csv( r"C:\Users\abhin\Desktop\Abhi\AOTM 2026\transportation_50x30_fixed_costs.csv")

fcost=df_fixed.iloc[0:50,1:31].values.tolist()
#print(fcost)

#sliced_data = df.iloc[start_index:end_index, start_column:end_column]
tcost=df.iloc[0:50,1:31].values.tolist()
#print(tcost)

demand = df.iloc[50,1:31].values.tolist()
print(demand)
supply = df.iloc[0:50, 31].values.tolist()
D=sum(supply)

#print(supply)

num_suppliers = len(supply)
num_cus = len(demand)
print("Number of suppliers:", num_suppliers)
print("Number of customers:", num_cus)

model=ConcreteModel()


# Define decision variables

model.x = Var(range(num_suppliers), range(num_cus), within=NonNegativeReals, doc="x")   
model.z = Var(range(num_suppliers), range(num_cus), within=Binary, doc="z")   


print('Number of variables =', len(model.x) + len(model.z))            
# Constraints

model.constraint1 = ConstraintList()

for i in range(num_suppliers):
    model.constraint1.add(sum(model.x[i, k] for k in range(num_cus)) == supply[i])

# Demand constraints
for k in range(num_cus):
    model.constraint1.add(sum(model.x[i, k] for i in range(num_suppliers)) == demand[k])

for i in range(num_suppliers):
    for k in range(num_cus):
        model.constraint1.add(model.x[i, k] <= min(supply[i], demand[k]) * model.z[i, k])  
        # model.constraint1.add(model.x[i, k] <= 100000000 * model.z[i, k])
        #min(supply[i], demand[k]) * model.z[i, k])

# Total number of constraints
total_constraints = len(model.constraint1)
print('Number of Constraints =', total_constraints)                

objective_terms = []
for i in range(num_suppliers):
    for k in range(num_cus):
        objective_terms.append(tcost[i][k] * model.x[i, k] + fcost[i][k] * model.z[i, k])

# Define the objective function
model.obj = Objective(expr=sum(objective_terms), sense=minimize)

solver = SolverFactory('cplex')  # or 'gurobi', 'cbc', 'highs', 'scip', 'glpk'

# solver.options['limits/time'] = 60 #scip

# solver.options['TimeLimit'] = 60 #gurobi


# SCIP
# solver.options['limits/time'] = 60
# solver.options['limits/gap'] = 0.05

# GUROBI
# solver.options['TimeLimit'] = 60
# solver.options['MIPGap'] = 0.05

# CPLEX
# solver.options['timeLimit'] = 60 
# solver.options['mipgap'] = 0.05


# CBC
# solver.options['sec'] = 60               
# solver.options['ratio'] = 0.05           



# HIGHS
# solver.options['time_limit'] = 60      
# solver.options['mip_rel_gap'] = 0.05   



start = time.perf_counter()

result = solver.solve(model, tee=True)
           
# solve_time = result.solver.time

end = time.perf_counter()
wall_time = end - start

print(f"Wall-clock solve time: {wall_time:.4f} seconds")
           
# if result.solver.status == SolverStatus.ok and result.solver.termination_condition == TerminationCondition.optimal:
if (result.solver.status == SolverStatus.ok and
    result.solver.termination_condition in
        (TerminationCondition.optimal,
         TerminationCondition.feasible,
         TerminationCondition.maxTimeLimit)):
    print('Time limit exceed')
    print('Total cost =', model.obj(), '\n')
    for i in range(num_suppliers):
        for k in range(num_cus):
            if model.x[i,k].value>0.01:
                print(f"x[{i+1},{k+1}] = {model.x[i, k].value:.2f}")
                # print(f"z[{i+1},{k+1}] = {model.z[i, k].value:.2f}")


