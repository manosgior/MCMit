# results

Consolidated CSV results feeding the paper's figures, named for what they
contain rather than for the script that happened to produce them.

| File | Feeds | Notes |
|---|---|---|
| `motivation_mcm_error_google.csv` | Fig. 3(a) — MCM error rate motivation | Copy of `lattice-sim/experiment_results/mcm_error_ler_google.csv` |
| `motivation_mcm_duration_google.csv` | Fig. 3(b) — MCM duration motivation | Copy of `lattice-sim/experiment_results/mcm_latency_ler_google.csv` |
| `qec_tradeoff_mcmit_ibm_current.csv` | Fig. 13(a) — IBM Heron, MCMit | Copy of `lattice-sim/experiment_results/mcm_tradeoff_mcm_ibm.csv` |
| `qec_tradeoff_herqules_ibm_current.csv` | Fig. 13(a) — IBM Heron, HERQULES | Copy of `lattice-sim/experiment_results/mcm_tradeoff_herqules_ibm.csv` |
| `qec_tradeoff_mcmit_futuristic.csv` | Fig. 13(b) — futuristic, MCMit | Copy of `lattice-sim/experiment_results/mcm_tradeoff_mcm_futuristic_2.csv` |
| `qec_tradeoff_herqules_futuristic.csv` | Fig. 13(b) — futuristic, HERQULES | Copy of `lattice-sim/experiment_results/mcm_tradeoff_herqules_futuristic_2.csv` |
| `qec_patches.csv` | Fig. 13(c) — QEC patches | Copy of `lattice-sim/experiment_results/mcm_patch_dist.csv` (the one the real pipeline actually invokes, not the older `mcm_patch.csv` variant) |
| `qec_threshold.csv` | Fig. 13(d) — QEC threshold | Copy of `lattice-sim/experiment_results/mcm_error_latency_ibm.csv` |
| `feedback_latency_impact_16q.csv` | Fig. 10, 16-qubit panels | Generated manually (no tracked script) |
| `feedback_latency_impact_32q.csv` | Fig. 10, 32-qubit panels | `evaluation/evaluate_decoherence.py` |
| `readout_duration_teleportation_fidelity.csv` | Fig. 12 — readout duration vs. teleportation fidelity | Generated manually (no tracked script) |
| `measurement_hardening_majority_voting.csv` | §7.2 — repetition-code / majority-voting fidelity | Per-shot fidelity with and without majority voting, GHZ, several qubit counts |
| `software_mitigation_fidelity.csv` | Fig. 11 — software mitigation fidelity | Real data, pulled from a remote server run |

The six `motivation_*`/`qec_*` files are the canonical Fig. 3/13 QEC data;
the actual generation pipeline (and its own README) lives in
[`lattice-sim/`](../lattice-sim/) — these are copies kept here so all
paper-result CSVs are in one place. If you regenerate them there, re-copy
into this directory.

`evaluation/generate_dummy_input.py` writes a *synthetic* stand-in
(`results/software_mitigation_fidelity_synthetic_test.csv`, not tracked
here) used to test the plotting pipeline before the real data above was
available — deliberately a different filename so re-running it can't
overwrite the real one.
