import pandas as pd

# Data for the CSV
data_circulant = {
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

}

# Creating the DataFrame
df_circulant = pd.DataFrame(data_circulant)

# Save to CSV
df_circulant.to_csv("Circulant_input_parameters_1.csv", index=False, quotechar='"')

print("CSV file created as 'Circulant_input_parameters_1.csv'.")

data_restart = {
    "RP0": [
        "0.1, 0.9, 0, 0, 0",
        "0.1, 0, 0.9, 0, 0",
        "0.1, 0, 0, 0.9, 0",
        "0.1, 0, 0, 0, 0.9",
        "0.1, 0, 0, 0, 0.9",
    ],
    "RP1": [
        "1, 0, 0, 0, 0",
        "1, 0, 0, 0, 0",
        "1, 0, 0, 0, 0",
        "1, 0, 0, 0, 0",
        "1, 0, 0, 0, 0",
    ],
    "RR0": ["0.9**1, 0.9**2, 0.9**3, 0.9**4, 0.9**5"] * 5,
    "RR1": ["0, 0, 0, 0, 0"] * 5,

}

# Creating the DataFrame
df_restart = pd.DataFrame(data_restart)

#Save to CSV
df_restart.to_csv("Restart_input_parameters_1.csv", index=False, quotechar='"')

print("CSV file created as 'Restart_input_parameters_1.csv'.")

# Data for the CSV
data_reversed = {
    "RCP0": [
        "0.5,0.5,0,0",
        "0,0.5,0.5,0",
        "0,0,0.5,0.5",
        "0.5,0,0,0.5",
    ],
    "CP1": [
        "0.5,0,0,0.5",
        "0.5,0.5,0,0",
        "0,0.5,0.5,0",
        "0,0,0.5,0.5",
    ],
    "CR0": ["-1,0,0,1"] * 4,
    "CR1": ["-1,0,0,1"] * 4,

}

# Creating the DataFrame
df_reversed = pd.DataFrame(data_reversed)

#Save to CSV
df_reversed.to_csv("Reversed_Circulant_input_parameters_1.csv", index=False, quotechar='"')

print("CSV file created as 'Reversed_Circulant_input_parameters_1.csv'.")


# Data for the CSV
data_mod_passive_circulant = {
    "MPCP0": [
        "0.5,0,0,0.5",
        "0.5,0.5,0,0",
        "0,0.5,0.5,0",
        "0,0,0.5,0.5",
    ],
    "MPCP1": [
        "0.5,0.5,0,0",
        "0,0.5,0.5,0",
        "0,0,0.5,0.5",
        "0.5,0,0,0.5",
    ],
    "MPCR0": ["-5,0,0,1"] * 4,
    "MPCR1": ["-1,0,0,1"] * 4,

}

# Creating the DataFrame
df_mod_passive_circulant = pd.DataFrame(data_mod_passive_circulant)

# Save to CSV
df_mod_passive_circulant.to_csv("Mod_passive_Circulant_input_parameters_1.csv", index=False, quotechar='"')

print("CSV file created as 'Mod_passive_Circulant_input_parameters_1.csv'.")


# Data for the CSV
data_mod_active_circulant = {
    "MACP0": [
        "0.5,0,0,0.5",
        "0.5,0.5,0,0",
        "0,0.5,0.5,0",
        "0,0,0.5,0.5",
    ],
    "MACP1": [
        "0.5,0.5,0,0",
        "0,0.5,0.5,0",
        "0,0,0.5,0.5",
        "0.5,0,0,0.5",
    ],
    "MACR0": ["-5,0,0,1"] * 4,
    "MACR1": ["-1,0,0,1"] * 4,

}

# Creating the DataFrame
df_mod_active_circulant = pd.DataFrame(data_mod_active_circulant)

# Save to CSV
df_mod_active_circulant.to_csv("Mod_active_Circulant_input_parameters_1.csv", index=False, quotechar='"')

print("CSV file created as 'Mod_active_Circulant_input_parameters_1.csv'.")
