import os
import numpy as np
import csv
import datetime
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD


# # Restart Input and Output file paths
# input_file = "Restart_input_parameters.csv"
# output_file = "Restart_output_results.txt"

# Circulant Input and Output file paths
input_file = "Circulant_input_parameters.csv"
output_file = "Circulant_output_results.txt"

folder = "results"

def read_parameters_from_csv(file_path):
    """Reads parameters from a CSV file and ensures data consistency."""
    P0, P1 = [], []
    R0, R1 = None, None
    alpha = None
    default_alpha = 0.2  # Default alpha value if not provided

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip the header row

        for row in reader:
            # Validate row length
            if len(row) < 4:
                raise ValueError(f"Incomplete row in CSV: {row}")

            try:
                # Parse RP0 and RP1
                P0.append([float(eval(value)) for value in row[0].split(",")])
                P1.append([float(eval(value)) for value in row[1].split(",")])

                # Parse Rr0 and Rr1 (ensure they are scalar values for each state)
                if R0 is None:
                    R0 = [float(eval(value)) for value in row[2].split(",")]
                if R1 is None:
                    R1 = [float(eval(value)) for value in row[3].split(",")]

                # Parse alpha (last column), if present
                if len(row) > 4 and row[4].strip():
                    alpha = float(eval(row[4]))
                else:
                    print(f"Warning: Missing alpha value in row {row}, using default alpha {default_alpha}")
                    alpha = default_alpha
            except Exception as e:
                raise ValueError(f"Error processing row {row}: {e}")

    # Convert RP0 and RP1 to NumPy arrays
    P0 = np.array(P0)
    P1 = np.array(P1)

    # Convert Rr0 and Rr1 to 1D NumPy arrays
    R0 = np.array(R0)
    R1 = np.array(R1)

    return P0, P1, R0, R1, alpha


def log_output(output, file_path):
    """Log output with timestamp to a text file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a') as logfile:
        logfile.write(f"[{timestamp}] {output}\n")


def save_output_to_csv(folder_path, base_filename, data, headers=None):
    """Saves results to a unique CSV file."""
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{base_filename}_{timestamp}.csv"
    file_path = os.path.join(folder_path, unique_filename)

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)

    print(f"Output saved to {file_path}")


def infinite_lp_index(P0, P1, R0, R1, alpha):
    """Calculate Whittle Indices using an LP relaxation approach."""
    n = len(R0)
    P = [P0, P1]
    R = [R0, R1]
    action = range(2)
    state = range(n)
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y", (action, state), lowBound=0.0, upBound=1.0)

    prob += lpSum([variables[1][s] for s in state]) == alpha
    for s in state:
        prob += variables[0][s] + variables[1][s] == lpSum([variables[a][ss] * P[a][ss][s] for a in action for ss in state])
    prob += lpSum([variables[a][s] for a in action for s in state]) == 1.0
    prob += lpSum([variables[a][s] * R[a][s] for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))

    gamma = list(prob.constraints.values())[0].pi
    T = 1000
    V = np.zeros((T + 1, n))
    Q = np.zeros((T, 2, n))
    I = np.zeros(n)

    for t in range(T):
        t = T - t - 1
        for a in action:
            for s in state:
                Q[t][a][s] = R[a][s] - a * gamma + sum(V[t + 1][ss] * P[a][s][ss] for ss in state)
        for s in state:
            V[t][s] = max(Q[t][0][s], Q[t][1][s])
    for s in state:
        I[s] = Q[0][1][s] - Q[0][0][s]
        if abs(I[s]) < 5e-7:
            I[s] = 0

    return list(np.argsort(-I)), I


def infinite_fix_point(P0, P1, R0, R1, alpha):
    """Calculate m* values using LP solution."""
    d = len(R0)
    P = [P0, P1]
    R = [R0, R1]
    V_1, V_2 = [], []

    action = range(2)
    state = range(d)
    prob = LpProblem("LP1", LpMaximize)
    variables = LpVariable.dicts("Y", (action, state), lowBound=0, upBound=1.0)

    prob += lpSum([variables[1][s] for s in state]) == alpha
    for s in state:
        prob += variables[0][s] + variables[1][s] == lpSum([variables[a][ss] * P[a][ss][s] for a in action for ss in state])
    prob += lpSum([variables[a][s] for a in action for s in state]) == 1.0
    prob += lpSum([variables[a][s] * R[a][s] for a in action for s in state])
    prob.solve(PULP_CBC_CMD(msg=1))

    m_star = np.zeros(d)
    for i in range(d):
        V1 = variables[0][i]
        V2 = variables[1][i]
        v1 = V1.varValue
        v2 = V2.varValue
        V_1.append(v1)
        V_2.append(v2)
        m_star[i] = v1 + v2

    return m_star, V_1, V_2


def process_bandit_problem():
    """Processes the restless bandit problem for given inputs."""
    try:
        RP0, RP1, Rr0, Rr1, alpha = read_parameters_from_csv(input_file)
        activation_states, indices = infinite_lp_index(RP0, RP1, Rr0, Rr1, alpha)
        log_output(f"Activation States: {activation_states}", output_file)
        log_output(f"Indices: {indices}", output_file)
        save_output_to_csv(folder, "indices", [[s, i] for s, i in zip(activation_states, indices)], ["State", "Index"])

        m_star, V_0, V_1 = infinite_fix_point(RP0, RP1, Rr0, Rr1, alpha)
        log_output(f"m*: {m_star}", output_file)
        save_output_to_csv(folder, "m_star", zip(m_star, V_0, V_1), ["m*", "V_0", "V_1"])

        print(f"Processing completed for {input_file}.")
    except Exception as e:
        log_output(f"Error: {str(e)}", output_file)
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    process_bandit_problem() # restart_input_file, restart_output_file, "Restart_results"
    # process_bandit_problem(circulant_input_file, circulant_output_file, "Circulant_results")
