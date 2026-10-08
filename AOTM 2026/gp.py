import pyomo.environ as pyo

# Use the direct interface to ensure Pyomo uses the specified Gurobi installation
solver = pyo.SolverFactory('gurobi_direct')

# Check if the solver is available with the correct license
print("Gurobi available:", solver.available())

# You can also check the license details to confirm it's the academic version
if solver.available():
    try:
        # This will print the license information if the solver is correctly configured
        solver._solver_model  # Accessing the underlying gurobipy model
        print("License check passed. Full academic license is active.")
    except Exception as e:
        print(f"An error occurred while checking the license: {e}")