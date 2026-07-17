# MCMit
Mid-circuit Measurement Error Mitigation

## Repository structure

| Path | What it is |
|---|---|
| `applications/` | The paper's benchmark circuits: constant-depth GHZ, long-range CNOT, teleportation, qubit reuse |
| `backends/` | QPU/simulator backend definitions and calibration data |
| `compiler/branching/` | Stochastic branching (§7.3) |
| `compiler/decoding/` | Measurement hardening — parity checks, repetition codes (§7.2) |
| [`compiler/removal/`](compiler/removal/README.md) | Static MCM elimination (§7.1), forked from [pcm-ccop-dc-pass](https://github.com/i2-tum/pcm-ccop-dc-pass) |
| [`qubit_state_discrimination/`](qubit_state_discrimination/README.md) | MCMit-CNN / MCMit-T discriminators and the HERQULES/QubiCML/baseline comparisons (§6, §8.3) |
| [`branch_instruction/`](branch_instruction/README.md) | The FPGA controller for `branch_reduce_fproc` (§5), forked from [QubiC](https://gitlab.com/LBL-QubiC) |
| [`lattice-sim/`](lattice-sim/README.md) | The QEC simulator behind Fig. 3 and Fig. 13 (§3.3, §8.5) |
| [`evaluation/`](evaluation/README.md) | Runnable per-figure evaluation drivers, plus [`evaluation/motivation/MECH/`](evaluation/motivation/MECH/README.md) for Fig. 2 (§3.2) |
| [`results/`](results/README.md) | Every figure's CSV data, in one place |
| `plotting/` | The remaining top-level plotting scripts (Fig. 10, 11, 12) |

## Dependencies

MCMit's own code (`applications/`, `backends/`, `compiler/`, `evaluation/`,
`plotting/`) needs:

- **Qiskit** (`qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`) — circuits, simulation, hardware access
- **mthree** — Qiskit M3 readout calibration, used by stochastic branching and GHZ evaluation
- **NumPy**, **pandas**, **NetworkX**, **matplotlib**, **seaborn** — numerics, the DAG compiler pass, plotting

Pinned in [`requirements.txt`](requirements.txt): `pip install -r requirements.txt`.

Each vendored subproject pins its own dependencies separately, since they're
independent tools with their own environments:
[`lattice-sim/requirements.txt`](lattice-sim/requirements.txt),
[`qubit_state_discrimination/requirements.txt`](qubit_state_discrimination/requirements.txt),
[`evaluation/motivation/MECH/requirements.txt`](evaluation/motivation/MECH/requirements.txt),
[`branch_instruction/qubic-software/pyproject.toml`](branch_instruction/qubic-software/pyproject.toml).

## External tools

MCMit builds on several external tools, either vendored with attribution
or reproduced as evaluation baselines. Each has its own README with setup
and reproduction details.

| Tool | What it is | How MCMit uses it |
|---|---|---|
| **lattice-sim** ([`lattice-sim/`](lattice-sim/README.md)) | Independent QEC simulator | Produces Fig. 3 and Fig. 13's surface-code logical-error-rate data (§3.3, §8.5); we added the MCM-latency sweep scripts and presets on top. Vendored with full commit history. |
| **MECH** ([`evaluation/motivation/MECH/`](evaluation/motivation/MECH/README.md)) | DQC chiplet compiler | Produces Fig. 2's sensitivity analysis (§3.2) via its own notebook, unmodified; we added a CSV-export script alongside it. |
| **HERQULES** ([`qubit_state_discrimination/`](qubit_state_discrimination/README.md)) | Matched-filter + FNN qubit-state discriminator | Reproduced as MCMit's primary discriminator baseline throughout §6/§8.3. |
| **QubiC** ([`branch_instruction/`](branch_instruction/README.md)) | Open-source FPGA-based QPU control framework | Forked and extended with the `branch_reduce_fproc` constant-latency multi-qubit branch instruction (§5). |
| **pcm-ccop-dc-pass** ([`compiler/removal/`](compiler/removal/README.md)) | Quantum-constant-propagation MCM-elimination compiler pass | Forked and extended with multi-qubit classical-control-logic simplification (§7.1). |

### References

- lattice-sim: S. Maurya and S. Tannu, "Synchronization for Fault-Tolerant
  Quantum Computers," ISCA 2025. https://doi.org/10.1145/3695053.3730991
  (artifact: https://zenodo.org/records/15092177)
- MECH: H. Zhang, K. Yin, A. Wu, H. Shapourian, A. Shabani, Y. Ding, "MECH:
  Multi-Entry Communication Highway for Superconducting Quantum Chiplets,"
  ASPLOS 2024. https://doi.org/10.1145/3620665.3640377
- HERQULES: S. Maurya, C. N. Mude, W. D. Oliver, B. Lienhard, S. Tannu,
  "Scaling Qubit Readout with Hardware Efficient Machine Learning
  Architectures," ISCA 2023. https://doi.org/10.1145/3579371.3589042
- QubiC: Y. Xu, G. Huang, J. Balewski, R. Naik, A. Morvan, B. Mitchell,
  K. Nowrouzi, D. I. Santiago, I. Siddiqi, "QubiC: An Open-Source
  FPGA-Based Control and Measurement System for Superconducting Quantum
  Information Processors," IEEE Transactions on Quantum Engineering,
  vol. 2, 2021. Source: https://gitlab.com/LBL-QubiC
- pcm-ccop-dc-pass: https://github.com/i2-tum/pcm-ccop-dc-pass, based on
  Y. Chen and Y. Stade, "Quantum Constant Propagation," in Static
  Analysis (SAS 2023), Springer, pp. 164–189, and extended in
  Y. Chen, I. Fulginiti, C. B. Mendl, "Reducing Mid-Circuit Measurements
  via Probabilistic Circuits," QCE 2024.
  https://ieeexplore.ieee.org/abstract/document/10821341, and
  "Optimization Framework for Reducing Mid-Circuit Measurements and
  Resets," Computational Science – ICCS 2025 Workshops, Springer.
  https://link.springer.com/chapter/10.1007/978-3-031-97570-7_13
