import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('results/software_mitigation_fidelity_placeholder.csv')

# Clean the data
df.columns = df.columns.str.strip()
df['Method'] = df['Method'].astype(str).str.strip()
df['Benchmark'] = df['Benchmark'].astype(str).str.strip()

# Get unique benchmarks
benchmarks = sorted(df['Benchmark'].unique())

over_qiskit = []
over_raw = []

for benchmark in benchmarks:
    print(f"\nAnalyzing {benchmark}:")
    
    # Filter data for this benchmark
    bench_data = df[df['Benchmark'] == benchmark]
    
    # Calculate improvements for each N
    improvements_mcmit_raw = []
    improvements_mcmit_m3 = []
    
    for n in sorted(bench_data['N'].unique()):
        n_data = bench_data[bench_data['N'] == n]
        
        mcmit_fid = n_data[n_data['Method'] == 'MCMit']['Fidelity'].values[0]
        raw_fid = n_data[n_data['Method'] == 'Raw']['Fidelity'].values[0]
        m3_fid = n_data[n_data['Method'] == 'Qiskit M3']['Fidelity'].values[0]
        
        improvements_mcmit_raw.append((mcmit_fid - raw_fid) / raw_fid * 100)
        improvements_mcmit_m3.append((mcmit_fid - m3_fid) / m3_fid * 100)
    
    # Calculate statistics
    avg_over_raw = np.mean(improvements_mcmit_raw)
    over_raw.append(avg_over_raw)
    max_over_raw = np.max(improvements_mcmit_raw)
    avg_over_m3 = np.mean(improvements_mcmit_m3)
    over_qiskit.append(avg_over_m3)
    max_over_m3 = np.max(improvements_mcmit_m3)
    
    print(f"MCMit vs Raw:")
    print(f"  Average improvement: {avg_over_raw:.2f}%")
    print(f"  Maximum improvement: {max_over_raw:.2f}%")
    print(f"MCMit vs Qiskit M3:")
    print(f"  Average improvement: {avg_over_m3:.2f}%")
    print(f"  Maximum improvement: {max_over_m3:.2f}%")
    
    
print(f"Average improvement Raw: {np.mean(over_raw):.2f}%")
print(f"Average improvement Qiskit: {np.mean(over_qiskit):.2f}%")