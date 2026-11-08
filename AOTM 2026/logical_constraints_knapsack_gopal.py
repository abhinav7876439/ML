
import numpy as np
from time import gmtime, strftime
import pyomo.environ as pyo

# -------------------------------------------------
# 1. Data generation (as provided)
# -------------------------------------------------

np.random.seed(1729)

print(
    'Date & Time now: ',
    strftime("%d-%m-%Y %H:%M:%S %Z%z", gmtime())
)

number_of_shipments = 500
weights_mu, weights_sigma = 70, 20 # weights of the shipments are randomly generated N(weights_mu, weights_sigma)
weights = abs(np.around(np.random.normal(weights_mu, weights_sigma, number_of_shipments),2)) # taken absolute values since weights cannot be negetive, rounded to second decimal places
volumes_mu, volumes_sigma = 10, 2 # volumes of the shipments are randomly generated N(volumes_mu, volumes_sigma)
volumes = abs(np.around(np.random.normal(volumes_mu, volumes_sigma, number_of_shipments),2)) # taken absolute values since weights cannot be negetive, rounded to second decimal places
values_mu, values_sigma = 50, 30 # values of the shipments are randomly generated N(values_mu, values_sigma)
values = abs(np.around(np.random.normal(values_mu, values_sigma, number_of_shipments),2)) # taken absolute since values of a shipment cannot be negetive, rounded to second decimal places

maximum_weight = 20500
maximum_volume = 3000

# -------------------------------------------------
# 2. Create the optimization model
# -------------------------------------------------

model = pyo.ConcreteModel()

# Set of shipments: 0, 1, ..., 499
model.I = pyo.RangeSet(0, number_of_shipments - 1)

# Binary decision variable
# x[i] = 1 if shipment i is selected, 0 otherwise
model.x = pyo.Var(model.I, domain=pyo.Binary)

# -------------------------------------------------
# 3. Objective function
# Maximize total shipment value
# -------------------------------------------------

model.objective = pyo.Objective(
    expr=sum(
        float(values[i]) * model.x[i]
        for i in range(number_of_shipments)
    ),
    sense=pyo.maximize
)

# -------------------------------------------------
# 4. Weight capacity constraint
# -------------------------------------------------

model.weight_constraint = pyo.Constraint(
    expr=sum(
        float(weights[i]) * model.x[i]
        for i in range(number_of_shipments)
    ) <= maximum_weight
)

# -------------------------------------------------
# 5. Volume capacity constraint
# -------------------------------------------------

model.volume_constraint = pyo.Constraint(
    expr=sum(
        float(volumes[i]) * model.x[i]
        for i in range(number_of_shipments)
    ) <= maximum_volume
)



# -------------------------------------------------
# 6. Logical constraints from Lecture 13 slides
# -------------------------------------------------

# # A selected => B selected
# model.constraint_A_B = pyo.Constraint(
#     expr=model.x[0] <= model.x[1]
# )

# # C and D cannot be selected together
# model.constraint_C_D = pyo.Constraint(
#     expr=model.x[2] + model.x[3] <= 1
# )

# # If E and F are both selected, G must be selected
# model.constraint_E_F_G = pyo.Constraint(
#     expr=model.x[4] + model.x[5] - 1 <= model.x[6]
# )

# # I, J and K cannot all be selected together
# model.constraint_I_J_K = pyo.Constraint(
#     expr=model.x[8] + model.x[9] + model.x[10] <= 2
# )

# # L and M can be selected together only if O and P are selected
# model.constraint_L_M_O = pyo.Constraint(
#     expr=model.x[11] + model.x[12] <= 1 + model.x[14]
# )

# model.constraint_L_M_P = pyo.Constraint(
#     expr=model.x[11] + model.x[12] <= 1 + model.x[15]
# )

# # Q is selected if and only if R is selected
# model.constraint_Q_R = pyo.Constraint(
#     expr=model.x[16] == model.x[17]
# )

# # At least one of S and T must be selected
# model.constraint_S_T = pyo.Constraint(
#     expr=model.x[18] + model.x[19] >= 1
# )
# -----------------------------------------------------------



# Constraint 1:
# If shipment 205 is selected, shipment 279 must be selected.
# model.logical_1 = pyo.Constraint(
#     expr=model.x[204] <= model.x[278]
# )

# # Constraint 2:
# # Shipments 326 and 123 cannot be selected together.
# model.logical_2 = pyo.Constraint(
#     expr=model.x[325] + model.x[122] <= 1
# )

# # Constraint 3:
# # Shipment 339 must be selected if both shipments 205 and 326 are selected.
# model.logical_3 = pyo.Constraint(
#     expr=model.x[204] + model.x[325] - 1 <= model.x[338]
# )

# # Constraint 4:
# # Shipments 221, 45, and 306 cannot all be selected together.
# model.logical_4 = pyo.Constraint(
#     expr=model.x[220] + model.x[44] + model.x[305] <= 2
# )

# # Constraint 5:
# # If shipments 352 and 187 are both selected, shipments 339 and 356 must also be selected.
# model.logical_5a = pyo.Constraint(
#     expr=model.x[351] + model.x[186] <= 1 + model.x[338]
# )

# model.logical_5b = pyo.Constraint(
#     expr=model.x[351] + model.x[186] <= 1 + model.x[355]
# )

# # Constraint 6:
# # Shipment 245 is selected if and only if shipment 205 is selected.
# model.logical_6 = pyo.Constraint(
#     expr=model.x[244] == model.x[204]
# )

# # Constraint 7:
# # At least one of shipments 279 and 366 must be selected.
# model.logical_7 = pyo.Constraint(
#     expr=model.x[278] + model.x[365] >= 1
# )


# -------------------------------------------------
# 6. Solve the model
# -------------------------------------------------

solver = pyo.SolverFactory('scip')

if not solver.available(exception_flag=False):
    raise RuntimeError(
        "SCIP solver not available. Install using: "
        "pip install pyscipopt"
    )

start_time = __import__('time').time()

results = solver.solve(model, tee=True)

end_time = __import__('time').time()

# -------------------------------------------------
# 7. Display results
# -------------------------------------------------

if results.solver.termination_condition == \
        pyo.TerminationCondition.optimal:

    selected = [
        i for i in range(number_of_shipments)
        if pyo.value(model.x[i]) > 0.5
    ]

    total_value = sum(values[i] for i in selected)
    total_weight = sum(weights[i] for i in selected)
    total_volume = sum(volumes[i] for i in selected)

    print("\n========== OPTIMAL SOLUTION ==========")
    print("Optimal total value:", round(total_value, 2))
    print("Number of selected shipments:", len(selected))
    print("Total weight:", round(total_weight, 2), "kg")
    print("Total volume:", round(total_volume, 2), "cu. ft.")
    print("Weight capacity:", maximum_weight, "kg")
    print("Volume capacity:", maximum_volume, "cu. ft.")
    print("Solution time:", round(end_time - start_time, 4), "seconds")

    print("\nSelected shipment details:")
    print("Shipment\tWeight\tVolume\tValue")

    for i in selected:
        print(
            "{}\t\t{:.2f}\t{:.2f}\t{:.2f}".format(
                i + 1, weights[i], volumes[i], values[i]
            )
        )

    # Save the selected shipment details
    import pandas as pd

    output = pd.DataFrame({
        'Shipment': [i + 1 for i in selected],
        'Weight': [weights[i] for i in selected],
        'Volume': [volumes[i] for i in selected],
        'Value': [values[i] for i in selected]
    })

    output.to_csv('selected_shipments.csv', index=False)

    print("\nSelected shipments saved to selected_shipments.csv")

else:
    print("Solver status:", results.solver.status)
    print(
        "Termination condition:",
        results.solver.termination_condition
    )
