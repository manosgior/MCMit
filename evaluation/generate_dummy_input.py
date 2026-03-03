import pandas as pd
import numpy as np
import os

# --- Configuration ---
csv_filename = 'results/dummy.csv'
benchmarks = {
    'Constant-depth GHZ': np.arange(5, 26, 2),            # Odd steps 5 to 25
    'Teleportation-Sequential': np.arange(5, 26, 2),   # Odd steps 5 to 25
    'Teleportation-Ladder': np.arange(1, 12, 1), # All steps 1 to 11
    'Long-range-CNOT': np.arange(5, 26, 2)         # Odd steps 5 to 25
}
methods = ['Raw', 'MCMit', 'Qiskit M3']
fidelity_mean = 0.5
fidelity_std_dev = 0.15 # Standard deviation for randomness

# --- Generate Data ---
data_rows = []

for benchmark_name, n_values in benchmarks.items():
    for n in n_values:
        for method_name in methods:
            # Generate random fidelity centered around mean, clip between 0 and 1
            fidelity = np.clip(np.random.normal(fidelity_mean, fidelity_std_dev), 0, 1)
            
            data_rows.append({
                'Benchmark': benchmark_name,
                'N': n,
                'Method': method_name,
                'Fidelity': fidelity
            })

# --- Create DataFrame and Save CSV ---
df_results = pd.DataFrame(data_rows)

# Ensure correct column order
df_results = df_results[['Benchmark', 'N', 'Method', 'Fidelity']]

# Save to CSV
df_results.to_csv(csv_filename, index=False)

print(f"Dummy data saved to '{csv_filename}'")
print("\nFirst few rows:")
print(df_results.head())
print("\nLast few rows:")
print(df_results.tail())