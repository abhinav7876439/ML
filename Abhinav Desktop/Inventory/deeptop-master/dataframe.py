import pandas as pd
import re

# === Step 1: Read policy text file ===
# Replace 'policy.txt' with your actual filename
with open("policy.txt", "r") as f:
    policy_text = f.read()

# === Step 2: Use regex to extract tuples and optimal actions ===
pattern = re.compile(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\):\s*(\d+)")
data = [tuple(map(int, match.groups())) for match in pattern.finditer(policy_text)]

# === Step 3: Convert to DataFrame ===
df = pd.DataFrame(data, columns=["x1", "x2", "x3", "x4", "optimal_action"])

# === Step 4: Save for verification ===
print(df.head())
print(f"\n✅ Total states parsed: {len(df)}")

# === Step 5: Optionally export to CSV or LaTeX ===
df.to_csv("policy_table.csv", index=False)
df.to_latex("policy_table.tex", index=False, caption="Optimal Policy Table", label="tab:policy", longtable=True)

print("\n📁 Files saved:")
print(" - policy_table.csv")
print(" - policy_table.tex")
