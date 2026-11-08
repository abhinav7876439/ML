import pandas as pd
import pyomo.environ as pyo

# ==========================================
# 1. READ DATA
# ==========================================
EXCEL_FILE = 'Case_Data.xlsx'
DISTANCE_SHEET = 'Distance_Matrix'
DEMAND_SHEET = 'demand'
RADIUS = 400
N_FACILITIES = 6   # Now valid with 486 cities

print(f"Reading data from {EXCEL_FILE}...")

df_dist_raw = pd.read_excel(EXCEL_FILE, sheet_name=DISTANCE_SHEET, header=None)
df_demand = pd.read_excel(EXCEL_FILE, sheet_name=DEMAND_SHEET, header=None)

# ==========================================
# 2. FIND THE DEMAND DATA AND DETERMINE N CITIES
# ==========================================
# Find the row where Sl. No. starts with 1
start_row_dem = None
for r in range(min(20, len(df_demand))):
    try:
        if int(float(df_demand.iloc[r, 0])) == 1:
            if int(float(df_demand.iloc[r+1, 0])) == 2:
                start_row_dem = r
                break
    except (ValueError, TypeError, IndexError):
        continue

if start_row_dem is None:
    raise ValueError("Could not locate the start of the demand data.")

# Count total number of cities (rows with valid integer IDs)
city_info = {}
for i in range(len(df_demand) - start_row_dem):
    row = df_demand.iloc[start_row_dem + i]
    try:
        city_id = int(float(row.iloc[0]))
        city_name = str(row.iloc[1]).strip()
        demand_val = float(row.iloc[3])
        if pd.isna(demand_val):
            break
        city_info[city_id] = {'name': city_name, 'demand': demand_val}
    except (ValueError, TypeError, IndexError):
        break

N_CITIES = len(city_info)
print(f"Detected {N_CITIES} cities in the demand data.")

if N_CITIES == 0:
    raise ValueError("No cities found. Check your Excel file layout.")

# ==========================================
# 3. FIND AND EXTRACT THE DISTANCE MATRIX
# ==========================================
# Find the header row (the row containing 1, 2, 3, ...)
start_row = None
start_col = None
for r in range(min(20, len(df_dist_raw))):
    for c in range(min(20, len(df_dist_raw.columns))):
        try:
            if (int(float(df_dist_raw.iloc[r, c])) == 1 and
                int(float(df_dist_raw.iloc[r, c+1])) == 2 and
                int(float(df_dist_raw.iloc[r, c+2])) == 3):
                start_row = r
                start_col = c
                break
        except (ValueError, TypeError, IndexError):
            continue
    if start_row is not None:
        break

if start_row is None:
    raise ValueError("Could not locate the distance matrix header.")

print(f"Detected matrix header at row {start_row}, column {start_col}")

# Extract N_CITIES x N_CITIES block
df_clean = df_dist_raw.iloc[
    start_row + 1 : start_row + 1 + N_CITIES,
    start_col     : start_col + N_CITIES
].copy()
df_clean = df_clean.apply(pd.to_numeric, errors='coerce').astype(float)

# Rename index and columns to 1..N_CITIES
df_clean.index = list(range(1, N_CITIES + 1))
df_clean.columns = list(range(1, N_CITIES + 1))

print(f"Extracted distance matrix shape: {df_clean.shape}")
print("="*40)

# ==========================================
# 4. BUILD A_ij MATRIX
# ==========================================
valid_ids = list(range(1, N_CITIES + 1))
demand_data = {cid: info['demand'] for cid, info in city_info.items()}

A_matrix = {}
for i in valid_ids:
    for j in valid_ids:
        A_matrix[(i, j)] = 1 if df_clean.loc[i, j] <= RADIUS else 0

# Quick stats
n_covered_pairs = sum(A_matrix.values())
print(f"Total (i,j) pairs within {RADIUS} km: {n_covered_pairs} / {N_CITIES**2}")
print("="*40)

# ==========================================
# 5. PYOMO MODEL
# ==========================================
model = pyo.ConcreteModel()
model.I = pyo.Set(initialize=valid_ids)
model.J = pyo.Set(initialize=valid_ids)
model.P = pyo.Param(model.I, initialize=demand_data, within=pyo.NonNegativeReals)
model.A = pyo.Param(model.I, model.J, initialize=A_matrix, within=pyo.Binary)
model.X = pyo.Var(model.I, domain=pyo.Binary)
model.Y = pyo.Var(model.J, domain=pyo.Binary)

def obj_rule(model):
    return sum(model.P[i] * model.X[i] for i in model.I)
model.Obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

def coverage_rule(model, i):
    return model.X[i] <= sum(model.A[i, j] * model.Y[j] for j in model.J)
model.CoverageConstraint = pyo.Constraint(model.I, rule=coverage_rule)

def num_facilities_rule(model):
    return sum(model.Y[j] for j in model.J) == N_FACILITIES
model.NumFacilitiesConstraint = pyo.Constraint(rule=num_facilities_rule)

# ==========================================
# 6. SOLVE
# ==========================================
solver = pyo.SolverFactory('cplex')
print(f"\nSolving with {N_FACILITIES} facilities out of {N_CITIES} candidates...")
results = solver.solve(model, tee=True)

# ==========================================
# 7. RESULTS
# ==========================================
print("\n" + "="*40)
print("OPTIMIZATION RESULTS")
print("="*40)

if results.solver.termination_condition == pyo.TerminationCondition.optimal:
    print(f"Total Market Potential Covered: {pyo.value(model.Obj):,.2f}\n")
    # Diagnostic: for each city, how many other cities can cover it?
    coverage_counts = [sum(A_matrix[(j, i)] for j in valid_ids) for i in valid_ids]
    print(f"Min coverage options per city: {min(coverage_counts)}")
    print(f"Max coverage options per city: {max(coverage_counts)}")
    print(f"Avg coverage options per city: {sum(coverage_counts)/len(coverage_counts):.2f}")
    print(f"Cities with ZERO coverage options: {sum(1 for c in coverage_counts if c == 0)}")


    print(f"Facilities to Open (Y_j = 1) [Total: {N_FACILITIES}]:")
    for j in model.J:
        if pyo.value(model.Y[j]) > 0.5:
            print(f"  -> {city_info[j]['name']}")

    covered = [i for i in model.I if pyo.value(model.X[i]) > 0.5]
    uncovered = [i for i in model.I if pyo.value(model.X[i]) < 0.5]
    print(f"\nCovered Cities: {len(covered)} / {N_CITIES}")
    print(f"Uncovered Cities: {len(uncovered)} / {N_CITIES}")

    total_demand = sum(demand_data.values())
    print(f"\nCoverage: {pyo.value(model.Obj):,.2f} / {total_demand:,.2f} = "
          f"{100*pyo.value(model.Obj)/total_demand:.2f}%")
else:
    print("No optimal solution found. Solver status:", results.solver.termination_condition)