import csv
import datetime
import markovianbandit as bandit
import os
import numpy as np

# Circulant Input and Output file paths
input_file = "Circulant_input_parameters.csv"
output_file = "Circulant_output_results.txt"

def read_parameters_from_csv(file_path):
    """Reads parameters from a CSV file and ensures data consistency."""
    CP0, CP1 = [], []
    CR0, CR1 = None, None
    alpha = None
    default_alpha = 0.7  # Default alpha value if not provided

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip the header row

        for row in reader:
            # Validate row length
            if len(row) < 4:
                raise ValueError(f"Incomplete row in CSV: {row}")

            try:
                # Parse CP0 and CP1
                CP0.append([float(value) for value in row[0].split(",")])
                CP1.append([float(value) for value in row[1].split(",")])

                # Parse CR0 and CR1 (ensure they are scalar values for each state)
                if CR0 is None:
                    CR0 = [float(value) for value in row[2].split(",")]
                if CR1 is None:
                    CR1 = [float(value) for value in row[3].split(",")]

                # Parse alpha (last column), if present
                if len(row) > 4 and row[4].strip():
                    alpha = float(row[4])
                else:
                    print(f"Warning: Missing alpha value in row {row}, using default alpha {default_alpha}")
                    alpha = default_alpha
            except Exception as e:
                raise ValueError(f"Error processing row {row}: {e}")

    # Convert CP0 and CP1 to NumPy arrays
    CP0 = np.array(CP0)
    CP1 = np.array(CP1)

    # Convert CR0 and CR1 to 1D NumPy arrays
    CR0 = np.array(CR0)
    CR1 = np.array(CR1)

    return CP0, CP1, CR0, CR1, alpha

def write_output_to_text_file(file_path, whittle_indices, alpha):
    """Write output to a text file."""
    with open(file_path, "a") as file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Date and Time: {current_time}\n")
        file.write(f"Alpha: {alpha}\n")
        file.write(f"Whittle Indices: {whittle_indices}\n\n")
    print(f"Output successfully written to text file: {file_path}")

def save_output_to_csv(folder_path, whittle_indices, alpha):
    """Save output to a CSV file."""
    os.makedirs(folder_path, exist_ok=True)  # Create directory if it doesn't exist

    # Prepare output file name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    output_file_path = os.path.join(folder_path, f"output_{timestamp}.csv")

    # Write the whittle indices and alpha to a CSV file
    with open(output_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Date and Time", "Alpha", "Whittle Indices"])
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([current_time, alpha, whittle_indices])

    print(f"Output successfully saved to {output_file_path}")

def main():
    try:
        # Read parameters from CSV file
        CP0, CP1, CR0, CR1, alpha = read_parameters_from_csv(input_file)

        # Log input shapes for debugging
        print(f"CP0 shape: {CP0.shape}, CP1 shape: {CP1.shape}")
        print(f"CR0 shape: {CR0.shape}, CR1 shape: {CR1.shape}")
        print(f"Alpha: {alpha}")

        # Initialize model and calculate Whittle indices
        model_circulant = bandit.restless_bandit_from_P0P1_R0R1(CP0, CP1, CR0, CR1)
        whittle_indices = model_circulant.whittle_indices() # discount=1
        

        # Write output to a text file
        write_output_to_text_file(output_file, whittle_indices, alpha)

        # Save output to a CSV file in a folder with date and time
        output_folder = "Circulant_output_results"
        save_output_to_csv(output_folder, whittle_indices, alpha)
        print(f"Whittle indices successfully calculated and saved.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
