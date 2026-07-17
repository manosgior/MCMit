# evaluation

Runnable drivers for the paper's evaluation, one entry point per figure.
Result CSVs land in [`results/`](../results/README.md); the pinned
dependencies in the top-level [`requirements.txt`](../requirements.txt)
are all these scripts need (the vendored subprojects below have their own).

Run everything from the repository root.

| Paper result | Command | Output |
|---|---|---|
| Fig. 2 — MECH motivation (§3.2) | `MECH_line_plot.ipynb` (last cell = paper 1×3 version) and `python motivation/MECH/export_fig2_csv.py` (run from `motivation/MECH/`) | `results/motivation_mech_*.csv`, `motivation/MECH/figures/mech_analysis.pdf` |
| Fig. 3 — QEC motivation (§3.3) | `python lattice-sim/evaluate_mcm_error.py --preset google` and `python lattice-sim/evaluate_mcm_latency.py --preset google`, plotted via `lattice-sim/plotting/mcm_plot.ipynb` — see [`lattice-sim/README.md`](../lattice-sim/README.md) | `results/motivation_mcm_*_google.csv` |
| Fig. 10 — feedback latency impact (§8.2) | `python -m evaluation.evaluate_feedback_latency --output results/feedback_latency_impact_16q.csv` (16q sim) and `python -m evaluation.evaluate_decoherence` (32q analytic) | `results/feedback_latency_impact_{16,32}q.csv` |
| Fig. 11 — software mitigation (§8.4) | `python -m evaluation.evaluate_software_mitigation --benchmark all --max-n 25` | `results/software_mitigation_fidelity.csv` schema |
| Fig. 12 — readout duration (§8.3) | `python -m evaluation.evaluate_readout_duration --output results/readout_duration_teleportation_fidelity.csv` | `results/readout_duration_teleportation_fidelity.csv` |
| Fig. 13 — QEC evaluation (§8.5) | `lattice-sim/` runner scripts + `lattice-sim/plotting/_render_four_panel.py` — see [`lattice-sim/README.md`](../lattice-sim/README.md) | `results/qec_*.csv` |
| Tables 3–6 — discriminators (§8.3) | `qubit_state_discrimination/runners/` — per-table commands in [`qubit_state_discrimination/README.md`](../qubit_state_discrimination/README.md); requires the readout-trace HDF5 dataset (not in the repo) | `optimization_reports/*.csv` |
| Table 8 / §7.1 analysis | `compiler/removal/` benchmark scripts (`mcm_count.py`, `branching_merged.py`, …) — see [`compiler/removal/README.md`](../compiler/removal/README.md) | stdout tables |

Figures are rendered from the CSVs by `plotting/plot.py` (Fig. 10/11/12) and
the notebooks/scripts inside `lattice-sim/plotting/` and
`evaluation/motivation/MECH/` (Fig. 2/3/13).

## Notes on fidelity vs. the published numbers

- **Fig. 11** (`evaluate_software_mitigation.py`): the published numbers were
  produced with this pipeline on **ibm_fez hardware** (10,000 shots, one
  calibration cycle). The script defaults to a noisy Aer simulator
  (FakeSherbrooke) so it runs anywhere — expect the same qualitative
  behaviour, not bit-identical values. For faithful results: run against
  real hardware and pass `--calibrations` a fresh M3 calibration of *that*
  backend (`compiler.branching.stochastic_branching.compute_calibrations_from_backend`);
  stochastic branching only helps when the confusion matrix matches the
  machine the circuits actually run on. `--noiseless` is a sanity mode
  (identity confusion matrix, everything should be ≈1.0 — verified).
  `--removal auto` skips stage 1 on the teleportation benchmarks, where
  upstream `ConstantPropagation.generate_instance` is not yet
  distribution-preserving (verified noiselessly; upstream pcm-ccop-dc-pass
  notes the instance-generation issue in its own `main.py`).
- **Fig. 10/12** (`evaluate_feedback_latency.py`, `evaluate_readout_duration.py`):
  thermal-relaxation-only models (T1=170 µs, T2=130 µs), matching the
  paper's "only model the thermal relaxation errors caused by qubit
  idling" setup; latencies default to Table 2's measured values and are
  CLI-configurable. Both circuit generators only support odd qubit counts,
  so "16-qubit" runs at 17.
