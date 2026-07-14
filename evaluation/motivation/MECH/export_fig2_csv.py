"""
export_fig2_csv.py
===================
Exports the raw data behind the paper's Fig. 2 (MCM error impact, MCM latency
impact, classical latency impact) as CSVs under results/, instead of just a
plot. Same computation as MECH_line_plot.ipynb's paper-version cell (the last
cell) -- run this after that notebook's cells 0-3 have produced
sample_exp_data/sample_baseline_data for chiplet_array_dim=(3,3) (or run this
script standalone; it reloads the same files itself).

Run from this directory: python3 export_fig2_csv.py
"""

import csv
import json
import os

baseline_mode = 'level_3'
baseline_data_path = 'sample_baseline_data'
exp_data_path = 'sample_exp_data'
structure = 'square'
chiplet_array_dim = (3, 3)
chiplet_size = (7, 7)
sparsity = None

results_dir = os.path.join('..', '..', '..', 'results')

additional_comment = '' if sparsity is None else f'_sparsity{sparsity}'
baseline_mode_comment = '_level_3' if baseline_mode == 'level_3' else ''
s = '_'.join(part[:3] for part in structure.split('_'))
baseline_file_name = f"{s}{chiplet_array_dim[0]}{chiplet_array_dim[1]}{chiplet_size[0]}{chiplet_size[1]}{additional_comment}{baseline_mode_comment}.json"
exp_file_name = f"{s}{chiplet_array_dim[0]}{chiplet_array_dim[1]}{chiplet_size[0]}{chiplet_size[1]}{additional_comment}.json"

with open(os.path.join(exp_data_path, exp_file_name)) as f:
    exp_data = json.load(f)
with open(os.path.join(baseline_data_path, baseline_file_name)) as f:
    baseline_data = json.load(f)[baseline_mode]

exp_depth_data, exp_on_chip_data, exp_cross_chip_data = {}, {}, {}
exp_meas_num_data, exp_shuttle_num_data = {}, {}
for b, data in exp_data.items():
    exp_depth_data[b] = data['depth']
    exp_on_chip_data[b] = data['on-chip']
    exp_cross_chip_data[b] = data['cross-chip']
    exp_meas_num_data[b] = data['meas_num']
    exp_shuttle_num_data[b] = data['shuttle_num']

baseline_depth_data, baseline_on_chip_data, baseline_cross_chip_data = {}, {}, {}
for b, data in baseline_data.items():
    baseline_depth_data[b] = data['depth']
    baseline_on_chip_data[b] = data['on-chip']
    baseline_cross_chip_data[b] = data['cross-chip']

benchmarks = list(exp_data.keys())

# --- (a) MCM error impact ---
cross_chip_weight = 10
meas_weights = [2, 4, 8, 10]
rows = []
for meas_w in meas_weights:
    for b in benchmarks:
        exp_eff_cnot = exp_on_chip_data[b] + cross_chip_weight * exp_cross_chip_data[b] + meas_w * exp_meas_num_data[b]
        baseline_eff_cnot = baseline_on_chip_data[b] + cross_chip_weight * baseline_cross_chip_data[b]
        rows.append((b, meas_w, 1 - exp_eff_cnot / baseline_eff_cnot))
with open(os.path.join(results_dir, 'motivation_mech_error_impact.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['benchmark', 'error_rate_ratio_mcm_2q', 'eff_2q_gates_difference'])
    w.writerows(rows)

# --- (b) MCM latency impact ---
meas_latencies = [3.6, 5, 10, 11, 20, 22, 30, 38, 40, 50]
rows = []
for meas_d in meas_latencies:
    for b in benchmarks:
        exp_depth = exp_depth_data[b] + exp_shuttle_num_data[b] * (meas_d - 2) * 2
        rows.append((b, meas_d, 1 - exp_depth / baseline_depth_data[b]))
with open(os.path.join(results_dir, 'motivation_mech_latency_impact.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['benchmark', 'latency_ratio_mcm_2q', 'depth_difference'])
    w.writerows(rows)

# --- (c) Classical latency impact ---
meas_latency = 10
classical_latencies = [2, 2.2, 3, 4, 5, 6, 7, 8, 10.3, 11]
rows = []
for c_l in classical_latencies:
    for b in benchmarks:
        exp_depth = exp_depth_data[b] + exp_shuttle_num_data[b] * (meas_latency - 2) * 2 + exp_meas_num_data[b] * c_l
        rows.append((b, c_l, 1 - exp_depth / baseline_depth_data[b]))
with open(os.path.join(results_dir, 'motivation_mech_classical_latency_impact.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['benchmark', 'latency_ratio_feedback_2q', 'depth_difference'])
    w.writerows(rows)

print('Wrote motivation_mech_{error,latency,classical_latency}_impact.csv to', results_dir)
