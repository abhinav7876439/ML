from pyomo.environ import *
import pandas as pd
import numpy as np
import re

# Read the Excel file
# df = pd.read_csv('transportation_50x30_balanced_merged.csv')

# df = pd.read_csv('transportation_50x30_balanced.csv',index_col=0)


# df = pd.read_csv( r"C:\Users\abhin\Desktop\Abhi\AOTM 2026\transportation_50x30_fixed_costs_merged.csv")


df = pd.read_csv( r"C:\Users\abhin\Desktop\Abhi\AOTM 2026\transportation_50x30_balanced.csv")

df_fixed_costs = pd.read_csv( r"C:\Users\abhin\Desktop\Abhi\AOTM 2026\transportation_50x30_fixed_costs.csv")



#sliced_data = df.iloc[start_index:end_index, start_column:end_column]
tcost=df.iloc[0:50,1:31].values.tolist()
# tcost=df.iloc[:-1,:-1].values.tolist()

fixed_costs = df_fixed_costs.iloc[0:50,1:31].values.tolist()


demand = df.iloc[50,1:31].values.tolist()
# demand = df.iloc[-1,:-1].values.tolist()
supply = df.iloc[0:50, 31].values.tolist()
D=sum(supply)
# supply = df.iloc[:-1, -1].values.tolist()

num_suppliers = len(supply)
num_cus = len(demand)
supplier_labels = df.iloc[0:num_suppliers, 0].astype(str).tolist()
customer_labels = df.columns[1:1 + num_cus].astype(str).tolist()
print("Number of suppliers:", num_suppliers)
print("Number of customers:", num_cus)

model=ConcreteModel()


# Define decision variables

model.x = Var(range(num_suppliers), range(num_cus), within=NonNegativeReals, doc="x") #Reals 
model.z = Var(range(num_suppliers), range(num_cus), within=Binary, doc="z")   # Integers

print('Number of variables =', len(model.x)+len(model.z))            
# Constraints

model.constraint1 = ConstraintList()

for i in range(num_suppliers):
    model.constraint1.add(sum(model.x[i, j] for j in range(num_cus)) == supply[i])

# Demand constraints
for j in range(num_cus):
    model.constraint1.add(sum(model.x[i, j] for i in range(num_suppliers)) == demand[j])

# Linking Constraints
for i in range(num_suppliers):
    for k in range(num_cus):
        model.constraint1.add(model.x[i, k] <= min(supply[i], demand[k]) * model.z[i, k])

# Total number of constraints
total_constraints = len(model.constraint1)
print('Number of Constraints =', total_constraints)                

objective_terms = []
for i in range(num_suppliers):
    for j in range(num_cus):
        objective_terms.append(tcost[i][j] * model.x[i, j] + fixed_costs[i][j] * model.z[i, j])

# Define the objective function
model.obj = Objective(expr=sum(objective_terms), sense=minimize)

solver = SolverFactory('highs')

result = solver.solve(model, tee=True)
           
if result.solver.status == SolverStatus.ok and result.solver.termination_condition == TerminationCondition.optimal:
    print('Optimal solution found')
    print('Total cost =', model.obj(), '\n')
    shipments = np.array([
        [value(model.x[i, j]) for j in range(num_cus)]
        for i in range(num_suppliers)
    ])
    shipments[np.abs(shipments) < 1e-6] = 0.0

    plan_df = pd.DataFrame(shipments, index=supplier_labels, columns=customer_labels)

    non_zero_routes = (
        plan_df.stack()
        .reset_index()
        .rename(columns={'level_0': 'Source', 'level_1': 'Destination', 0: 'Quantity'})
    )
    non_zero_routes = non_zero_routes[non_zero_routes['Quantity'] > 0.01]

    print('\nNon-zero shipping routes:')
    if non_zero_routes.empty:
        print('No positive shipments found.')
    else:
        print(non_zero_routes.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    plan_df.to_csv('transportation_plan_matrix.csv', float_format='%.2f')
    non_zero_routes.to_csv('transportation_plan_nonzero.csv', index=False, float_format='%.2f')
    print('\nSaved detailed plan files: transportation_plan_matrix.csv and transportation_plan_nonzero.csv')

    supply_check = pd.DataFrame(
        {
            'Supply': supply,
            'Shipped': plan_df.sum(axis=1).values
        },
        index=supplier_labels
    )
    demand_check = pd.DataFrame(
        {
            'Demand': demand,
            'Received': plan_df.sum(axis=0).values
        },
        index=customer_labels
    )

    print('\nSupply balance check (Supply vs Shipped):')
    print(supply_check.to_string(float_format=lambda x: f"{x:,.2f}"))

    print('\nDemand balance check (Demand vs Received):')
    print(demand_check.to_string(float_format=lambda x: f"{x:,.2f}"))


