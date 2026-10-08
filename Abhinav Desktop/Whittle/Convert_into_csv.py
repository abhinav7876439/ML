import pandas as pd

# Data for the CSV
data = {
    "CP0": [
        "0.5,0,0,0.5",
        "0.5,0.5,0,0",
        "0,0.5,0.5,0",
        "0,0,0.5,0.5",
    ],
    "CP1": [
        "0.5,0.5,0,0",
        "0,0.5,0.5,0",
        "0,0,0.5,0.5",
        "0.5,0,0,0.5",
    ],
    "CR0": ["-1,0,0,1"] * 4,
    "CR1": ["-1,0,0,1"] * 4,

    "alpha": 0.2
}

# Creating the DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("Circulant_input_parameters_1.csv", index=False, quotechar='"')

print("CSV file created as 'Circulant_input_parameters_1.csv'.")


# Data for the CSV
data_non_whittle = {
    "NWP0": [
        "0.005, 0.793, 0.202",
        "0.027, 0.558 , 0.415",
        "0.736, 0.249, 0.015",
    ],
    "NWP1": [
        "0.718, 0.254, 0.028",
        "0.347, 0.097, 0.556",
        "0.015, 0.956, 0.029",
    ],
    "NWR0": ["0, 0, 0"] * 3,
    "NWR1": ["0.699, 0.362, 0.715"] * 3,

    "alpha": 0.2
}

# Creating the DataFrame
df_non_whittle = pd.DataFrame(data_non_whittle)

# Save to CSV
df_non_whittle.to_csv("non_whittle_input_parameters_1.csv", index=False, quotechar='"')

print("CSV file created as 'non_whittle_input_parameters_1.csv'.")

# Data for the CSV
df_active_circulant = {
    "P0": [
        "0.5,0,0,0.5",
        "0.5,0.5,0,0",
        "0,0.5,0.5,0",
        "0,0,0.5,0.5",
    ],

    "P1": [
        "0.5,0.5,0,0",
        "0,0.5,0.5,0",
        "0,0,0.5,0.5",
        "0.5,0,0,0.5",
    ],
    "R0": ["-1,0,0,5"] * 4,
    "R1": ["-1,0,0,5"] * 4,

    "alpha": 0.2
}

# Creating the DataFrame
df_active_circulant = pd.DataFrame(df_active_circulant)

# Save to CSV
df_active_circulant.to_csv("Active_passive_input_parameters_1.csv", index=False, quotechar='"')

print("CSV file created as 'Active_passive_input_parameters_1.csv'.")
