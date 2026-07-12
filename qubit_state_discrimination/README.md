# qubit_state_discrimination

This directory implements deep-learning and signal-processing classifiers for
**qubit state discrimination** in superconducting quantum computers.  It covers the
full HERQULES pipeline (matched-filter pre-processing + neural network classifier),
the paper's own MCMit-CNN and MCMit-T (transformer) discriminators, reproductions of
QubiCML and a raw-trace baseline FNN, and KLiNQ — a knowledge-distillation pipeline
designed for FPGA deployment.

This directory is synced from the active research workspace at `Oraqle/Discriminators/`
(kept outside this repo). If you're looking for what actually produces the paper's
Tables 3–6, start at [Reproducing the MCMit Paper's Discriminator Tables](#reproducing-the-mcmit-papers-discriminator-tables--83).

---

## Reproducing the MCMit Paper's Discriminator Tables (§ 8.3)

Covers the discriminator-comparison tables (Table 3: long-trace accuracy for all 5
designs; Table 4: accuracy vs. trace length for HERQULES/QubiCML/MCMit-CNN), the
cross-fidelity/crosstalk table (Table 5), and the accuracy-vs-simultaneous-qubits table
(Table 6).

**Environment.** These scripts are written to run inside the Docker image built from
Oraqle's `Dockerfile`, with `/data/five_qubit_data` (raw HDF5 dataset) and
`/app/optimization_reports` (output CSVs) bind-mounted in — the absolute paths are
hardcoded in each script's header. `data/five_qubit_data/` and `data/single_qubit_data/`
in this checkout are placeholders, not the real dataset (see
[Disclaimer & Research Context](#disclaimer--research-context)); nothing below was run to
verify numbers in this environment (no GPU/dataset available), only statically
cross-checked against the code.

### Design → code mapping

| Paper design | Network | Trainer / pipeline |
|---|---|---|
| Baseline FNN (Lienhard et al. [62]) | [`networks/SingleQubitFNN.py`](networks/SingleQubitFNN.py)`::SingleQubitFNN_Baseline` | `runners/hyper_optimize.py` |
| QubiCML (Vora et al., arXiv:2406.18807 [97]) | [`networks/Qubic.py`](networks/Qubic.py)`::Arxiv240618807FNN` | `runners/hyper_optimize.py` |
| HERQULES (Maurya et al., ISCA'23 [68]) | [`networks/HERQULES.py`](networks/HERQULES.py)`::Net_rmf` | [`trainers/HERQULES_original.py`](trainers/HERQULES_original.py) |
| MCMit-T (transformer) | [`networks/TransformerMF.py`](networks/TransformerMF.py)`::QubitClassifierTransformerMF` | `runners/hyper_optimize_transformer_mf.py` |
| MCMit-CNN | [`networks/CNN.py`](networks/CNN.py)`::CNN` | `runners/_colleague_prep.py` (preprocessing) + `runners/_cnn_length_sweep.py` / `runners/_xtalk_cnn.py` |

### Table 3 — accuracy at full 1µs trace, all 5 designs

| Row | Command | Output |
|---|---|---|
| Baseline FNN, QubiCML | `python runners/hyper_optimize.py` (500-sample length; calls `optimize_models(["FNN"])` and `optimize_models(["Arxiv240618807FNN"])`) | `optimization_reports/*.csv` |
| HERQULES | `python runners/_xtalk_herqules_deploy.py` (prints `F5Q (sanity) 0.905`); for a full per-qubit CSV row use `python runners/_herqules_length_both.py` at the `ns=1000` row | `xtalk_HERQULES_deployment.csv` / `herqules_length_faithful_vs_demux.csv` |
| MCMit-T | `python runners/hyper_optimize_transformer_mf.py` | `optimization_reports/*.csv` |
| MCMit-CNN | `CNN_LEN=500 python runners/_cnn_length_sweep.py` | `optimization_reports/cnn_length_1000ns.csv` |

No script currently assembles all 5 rows into one table — merge the CSVs by hand, or
extend `fetch_results.py` / `plot_fidelities.py`'s `master_fidelity.csv` schema.

### Table 4 — accuracy vs. trace length (250 / 500 / 750 ns)

| Row | Command |
|---|---|
| HERQULES | `python runners/_herqules_truncation_pertrace.py` (`LENGTHS_NS` already spans 250/500/750; filter rows to `policy=="freeze"` — the no-retrain/frozen-network policy, which reproduces the paper's near-chance HERQULES@250ns numbers) |
| QubiCML | no dedicated script — reuse `runners/hyper_optimize.py` with `TRACE_LENGTHS=[125,250,375]` (samples for 250/500/750ns) |
| MCMit-CNN | `CNN_LEN={125,250,375} python runners/_cnn_length_sweep.py` (one run per length) |

`runners/_herqules_truncation_eval.py` is a near-identical alternative on the
paper-subsampled (3000/7000 shots/class) data rather than the full dataset — use
`_herqules_truncation_pertrace.py` to stay consistent with the full-data numbers used
everywhere else in this table.

### Table 5 (cross-fidelity by qubit separation) and Table 6 (accuracy vs. simultaneously-measured qubits N)

Both come from the **same run** of two scripts — each computes the crosstalk
cross-fidelity metric *and* the accuracy-vs-N sweep (geomean accuracy over all `C(5,N)`
driven-qubit subsets, other qubits held at ground state) in one pass:

| Design | Command | Output rows |
|---|---|---|
| HERQULES | `python runners/_xtalk_herqules_deploy.py` | `xtalk_HERQULES_deployment.csv`: `metric=="crossfid_dist"` → Table 5, `metric=="F_vs_N"` → Table 6 |
| MCMit-CNN | `python runners/_xtalk_cnn.py` | `xtalk_CNN.csv`: same `metric` column |

### Caveats — read before trusting a number

This sync was done by statically reading the code; nothing here was executed to
cross-check against the paper's published values (no GPU/dataset in this environment):

- **HERQULES has two parallel implementations.** `trainers/train_HERQULES.py` ("faithful
  replication," demux-style MF features) gives ~0.925 F5Q; `trainers/HERQULES_original.py`
  (the original-paper port) with **per-trace, deployment-style** features gives ~0.904–0.905
  F5Q — the one matching Table 3. Scripts named `*_deploy*`/`*_pertrace*` use the latter;
  keep any new script on that path too.
  [`matched_filter.py`](matched_filter.py)`::matched_filter_preprocess_demux` has a
  `scramble` flag: `scramble=False` (realistic — a feature row's 5 qubit MF scores come
  from the *same* physical shot) is what the `*_deploy*`/`*_pertrace*` scripts use;
  `scramble=True` reproduces the original per-(qubit,state) independently-permuted draw.
- **MCMit-CNN has two preprocessing paths.** `helpers/cnn_helpers.py` (older, ~0.908 F5Q)
  vs. `runners/_colleague_prep.py` (FFT-based per-qubit frequency/phase calibration +
  windowed-sinc sparse FIR demod, matching a reference implementation; ~0.911 F5Q, matches
  Table 3). `_cnn_length_sweep.py` and `_xtalk_cnn.py` both use `_colleague_prep.py` — don't
  substitute `cnn_helpers.py` in for these tables, even though `hyper_optimize.py` still
  uses it for the FNN/QubiCML rows.
- [`networks/HybridCNN.py`](networks/HybridCNN.py) and its `_xtalk_hybridcnn.py` /
  `train_hybrid_cnn.py` scripts are a separate, earlier CNN design — not the one branded
  MCMit-CNN in the paper. Kept for reference only; not part of Tables 3–6.
- `train_arxiv_model.py` (root) has a stale import (`networks.Arxiv240618807FNN`, renamed
  to `networks/Qubic.py` upstream) and is broken as committed — both here and in the
  upstream workspace. Use `runners/hyper_optimize.py` for QubiCML instead.
- `runners/train_three_level_*.py`, `assemble_three_level_table.py`, and
  `reproduce_klinq_paper.py` belong to a *different* survey (qutrit/|2⟩-leakage
  three-level discrimination) and were **not** copied into this checkout.

Table 7 (FPGA resource utilization via hls4ml) is not covered by this sync; the relevant
scripts (`_cnn_shallow_hls.py`, `_cnn_small_hls.py`, `_debug_synth.py`) still live only in
`Oraqle/Discriminators/`.

---

## Table of Contents

0. [Reproducing the MCMit Paper's Discriminator Tables (§ 8.3)](#reproducing-the-mcmit-papers-discriminator-tables--83)
1. [Physical Background](#physical-background)
2. [HERQULES Pipeline](#herqules-pipeline)
   - [Overview and Design Philosophy](#overview-and-design-philosophy)
   - [Stage 1 — Frequency Demodulation](#stage-1--frequency-demodulation)
   - [Stage 2 — Pre-classification and Trace Purification](#stage-2--pre-classification-and-trace-purification)
   - [Stage 3 — Matched Filter and Relaxation Matched Filter](#stage-3--matched-filter-and-relaxation-matched-filter)
   - [Stage 4 — Neural Network Classifier](#stage-4--neural-network-classifier)
3. [Matched Filter Module](#matched-filter-module)
4. [Additional Model Architectures](#additional-model-architectures)
   - [1. QubiCML (arXiv:2406.18807 FNN)](#1-qubicml-arxiv240618807-fnn)
   - [2. Transformer (QubitClassifierTransformer / MF variant)](#2-transformer-qubitclassifiertransformer--mf-variant)
   - [3. MCMit-CNN](#3-mcmit-cnn)
   - [4. KLiNQ — Knowledge Distillation Pipeline](#4-klinq--knowledge-distillation-pipeline)
5. [Repository Structure](#repository-structure)
6. [Data Pipeline](#data-pipeline)
7. [Usage Instructions](#usage-instructions)
8. [Development Story of KLiNQ](#development-story-of-klinq)
9. [Disclaimer & Research Context](#disclaimer--research-context)

---

## Physical Background

In dispersive qubit readout a microwave tone is sent through a resonator coupled to the
qubit.  The qubit state (|0⟩ or |1⟩) shifts the resonator frequency, producing a
characteristic phase and amplitude response.  The signal is **downconverted** to an
intermediate frequency (IF) and digitised by an ADC, yielding two quadrature components:

- **I** (in-phase)
- **Q** (quadrature)

For **multiplexed readout** of 5 qubits, all five resonators are probed simultaneously
at different IF frequencies in a shared bandwidth.  Dataset files contain raw IQ traces
of shape `(N_shots, trace_length, 2)` and integer labels in `[0, 31]` (one bit per qubit,
bit *k* encodes qubit *k+1*).

Dataset hardware parameters:

| Parameter | Value |
|---|---|
| Number of qubits | 5 |
| ADC sampling rate | 500 MHz |
| Readout window | 1 µs (500 samples) |
| IF frequencies | −64.73, −25.37, +24.79, +70.27, +127.28 MHz |

---

## HERQULES Pipeline

**File:** [`HERQULES.py`](HERQULES.py)

HERQULES (**H**ierarchical **E**fficient **R**eadout with **QU**bit **L**earning via
**E**nsemble **S**tages) is the primary classification pipeline in this repository.
It combines signal-processing and machine-learning stages to achieve accurate multi-qubit
state discrimination from the raw IQ traces, using a compact feature set designed to be
deployable on an FPGA.

### Overview and Design Philosophy

Raw multiplexed IQ traces contain all 5 qubit signals superimposed at different IF
frequencies.  Direct classification from the raw trace requires a large model and gives
no interpretability.  HERQULES instead uses a structured 4-stage pipeline:

```
Raw multiplexed IQ traces
         │
         ▼  Stage 1
Frequency Demodulation          ← per-qubit digital down-conversion + LPF
         │
         ▼  Stage 2
Pre-classification / Purification  ← geometric IQ clustering; label clean traces;
                                      catalogue error events (relax, |2⟩, excite)
         │
         ▼  Stage 3
Matched Filter (MF)  +  Relaxation MF (RMF)
  ← learn optimal linear envelope per qubit for |0⟩ vs |1⟩
  ← learn optimal linear envelope per qubit for relax vs |0⟩
         │
         ▼  Stage 4
Compact MLP (Net_rmf: 10→10→20→32)
  ← input: 5 MF scalars + 5 RMF scalars
  ← output: logits over 32 basis states
```

This pipeline yields a **10-dimensional** feature vector (5 MF + 5 RMF scalars) which is
fed into a tiny MLP.  The entire forward pass at inference time involves only a few
hundred multiply-accumulate operations — highly FPGA-friendly.

### Stage 1 — Frequency Demodulation

**Function:** `demodulate_multiplexed_traces(iq_traces, qubit_frequencies, sampling_rate, ...)`

The raw multiplexed IQ stream is processed per qubit:

1. Optional DC-offset removal and I/Q amplitude imbalance correction.
2. **Digital down-conversion**: multiply the IQ phasor by `exp(j 2π f_IF t)` to shift
   the target resonator's contribution to DC.
3. **Low-pass filter** (3rd-order Butterworth, default cut-off 10 MHz) to suppress all
   other resonator signals.
4. Save per-qubit IQ traces to `demodulated_q{k}_.h5`.

After demodulation, each qubit's traces are shape `(N_shots, 500, 2)`.

### Stage 2 — Pre-classification and Trace Purification

**Class:** `preclassifier`  
**Helper:** `get_traces(num_qubits, plot, rscale, data_type)`

Before computing matched-filter envelopes it is critical to build *clean* reference
traces for each qubit state.  `preclassifier` clusters the time-averaged IQ responses
into |0⟩ and |1⟩ clouds.

For each qubit:
1. Compute the IQ centroid of each cloud.
2. Define a *purity radius* = `rscale × (inter-centroid distance / 2)`.
3. Keep only traces within the radius as clean reference traces.
4. Classify traces *outside* the radius into three error categories:

| Error category | Criterion |
|---|---|
| **Relaxation** (|1⟩→|0⟩) | |1⟩-labelled trace whose IQ mean falls inside the |0⟩ cluster |
| **|2⟩ leakage** | |1⟩-labelled trace outside both clusters |
| **Thermal excitation** (|0⟩→|1⟩) | |0⟩-labelled trace outside the |0⟩ cluster |

The classifier state (indices and trace classes) is persisted to `preclassifier_state.pkl`
and loaded back for reuse.

Key methods:

| Method | Description |
|---|---|
| `fit()` | Run the purification pipeline; populate `filtered_indices` and `trace_classes`. |
| `predict(data)` | Extract purified traces from a raw array using stored indices. |
| `save_state(filename)` | Pickle the classifier state. |
| `load_state(filename)` | Restore classifier state from pickle. |
| `get_traces()` | Return the per-qubit trace-class dictionary. |

### Stage 3 — Matched Filter and Relaxation Matched Filter

**Class:** `relaxation_mf_classifier`  
**Module:** [`matched_filter.py`](matched_filter.py)  
**Helper:** `get_mf(traces_0, traces_1)`

Two independent matched filters are computed per qubit:

#### Standard Matched Filter (MF)

Discriminates |0⟩ vs |1⟩.  The Wiener-optimal linear envelope is:

```
h_MF = E[x_0 − x_1] / Var[x_0 − x_1]
```

A **boxcar window** is multiplied with the envelope to truncate integration at a per-qubit
optimal time, reducing sensitivity to late-time noise.  The boxcar widths used in practice
are `[1, 1, 9, 2, 9]` (in units of 50 ADC samples).

The discrimination threshold is set at the 99.5th percentile of the MF output distribution
on |0⟩ traces, providing a high-confidence acceptance region.

#### Relaxation Matched Filter (RMF)

Distinguishes clean |0⟩ traces from |1⟩→|0⟩ *relaxation* traces:

```
h_RMF = E[x_relax − x_0] / Var[x_relax − x_0]
```

This filter has a characteristic shape that reflects the IQ trajectory of a qubit that
started in |1⟩ but decayed to |0⟩ during the readout window.  It provides complementary
information to the standard MF and significantly improves discrimination accuracy in the
presence of qubit relaxation.

Key methods of `relaxation_mf_classifier`:

| Method | Description |
|---|---|
| `fit(trace_classes, num_qubits, boxcars)` | Compute per-qubit RMF envelopes and thresholds. |
| `predict(num_qubits, data_type, trace_length)` | Apply RMF to train/val/test data. |
| `save_state(filename)` | Pickle envelopes, thresholds, and inherited state. |
| `load_state(filename)` | Restore from pickle. |

### Stage 4 — Neural Network Classifier

**Classes:** `Net` (MF-only) and `Net_rmf` (MF + RMF, primary)

The compact MLP takes the concatenated MF and RMF scalar outputs (10 features) and
produces logits over 32 classes:

```
Input (10) → Linear(10→10) → ReLU → Linear(10→20) → ReLU → Linear(20→32)
```

A `CrossEntropyLoss` with Adam (lr = 0.01) is used.  A step-decay LR schedule divides
the learning rate by 10 at epochs 30, 60, and 90.  Training runs for 100 epochs and the
best checkpoint (by validation accuracy) is saved to `checkpoints/mf_nn/best_epoch.pth`.

There is also a **baseline** MLP (`Net_baseline`) that takes the full 1000-sample raw
trace as input (1000 → 500 → 250 → 32).  It serves as an accuracy upper-bound but is not
suitable for FPGA deployment.

#### Entry Points

| Function | Description |
|---|---|
| `train(run_pre_filter, run_semi_sup, run_rmf, dur)` | Full HERQULES pipeline. |
| `test()` | Evaluate `Net_rmf` from saved checkpoint. |
| `train_baseline()` | Train the raw-trace baseline MLP. |

---

## Matched Filter Module

**File:** [`matched_filter.py`](matched_filter.py)

This module implements the matched-filter computations used in Stage 3.  It is imported
by `HERQULES.py` but can also be used independently.

### Core Functions

| Function | Description |
|---|---|
| `MF_meas(X_train, X_test, y_train, y_test, stop_index, bcub, ...)` | Single matched-filter train + evaluate (binary, with optional boxcar and SVM threshold). |
| `MF_SVM_limit(X, y)` | SVM-based threshold optimisation: fits a LinearSVC on 1-D MF outputs and finds the decision boundary. |
| `MF_single_disc(X, y, stop_index, th_limit_C)` | Searches for the optimal boxcar width by iteratively shortening the integration window. |
| `obtain_matched_filter_with_bcub(X, y, stop_index, th_limit_C, best_bc)` | Compute MF envelope with a fixed pre-determined boxcar. |
| `find_best_matched_filter(train_gnd, train_ext, best_bc)` | High-level wrapper: constructs train arrays, calls disc or bcub depending on `best_bc`. |

### Multi-qubit Helpers

| Function | Variant | Description |
|---|---|---|
| `search_matched_filter_for_all_qubits` | Standard | Compute MF for all 5 qubits from a stacked `(32, N, T, 2)` array. |
| `search_matched_filter_for_all_qubits_demux` | Demux | Same, but expects per-qubit data list from `mf_demux_data_prep`. |
| `search_matched_filter_for_all_qubits_preclass` | Pre-class | Same, but selects ground/excited traces by label from a flat purified array. |

### Preprocessing and Evaluation

| Function | Description |
|---|---|
| `matched_filter_preprocess(data, envelopes)` | Apply 5 MF envelopes to stacked data → `(32, N_min, 5)` scalar array. |
| `matched_filter_preprocess_demux(data, envelopes)` | Same but for per-qubit demux structure. |
| `calculate_matched_filter_acc(data, all_mfs, all_thres)` | Evaluate MF classifier on stacked data; print accuracy. |
| `calculate_matched_filter_acc_demux(data, all_mfs, all_thres)` | Same for demux data; also saves `mf_preds.pkl`. |

### Matched Filter Mathematics

For a binary |0⟩ / |1⟩ task the matched filter envelope is:

```
h = E[x_0 − x_1] / Var[x_0 − x_1]
```

where `x_0`, `x_1` are flattened IQ traces (I followed by Q).  Classification:

```
y_pred = (x · h) < threshold
```

The *boxcar* is a rectangular window applied element-wise: it zeros the envelope beyond
a given time index, restricting the integration window.

---

## Additional Model Architectures

### 1. QubiCML (arXiv:2406.18807 FNN)

**File:** [`networks/Qubic.py`](networks/Qubic.py) (class `Arxiv240618807FNN`; file renamed
from `Arxiv240618807FNN.py` — same file, same class name, new module path)  
**Training script:** `runners/hyper_optimize.py` (`optimize_models(["Arxiv240618807FNN"])`).
`train_arxiv_model.py` (root) still imports the *old* `networks.Arxiv240618807FNN` path and
is broken as committed — don't use it, see the [caveats above](#caveats--read-before-trusting-a-number).

This is the paper's "QubiCML" baseline: a reproduction of the lightweight FNN described in
[arXiv:2406.18807](https://arxiv.org/abs/2406.18807) (Vora et al.).

#### Architecture

```
Input (2)  ──► Linear(2→8) ──► ReLU ──► Linear(8→4) ──► ReLU ──► Linear(4→1) ──► Sigmoid
```

| Layer | Size | Activation |
|---|---|---|
| Input | 2 | — |
| Hidden 1 | 8 | ReLU |
| Hidden 2 | 4 | ReLU |
| Output | 1 | Sigmoid |

#### Input Processing

The raw 5-qubit multiplexed trace is **demodulated** per qubit before being fed to the network:

1. **Frequency demodulation** — for each of the 5 qubit IF frequencies, rotate the trace by
   `exp(j 2πf_IF t)` and integrate over the readout window.
2. **Min-max normalisation** — scale each (I, Q) column to [0, 1].
3. **Per-qubit labelling** — `y_q = (y >> q) & 1`.
4. **One model per qubit** — 5 separate `Arxiv240618807FNN` instances.

#### Training Configuration

| Parameter | Value |
|---|---|
| Loss | Binary Cross-Entropy (`nn.BCELoss`) |
| Optimiser | Adam (lr = 1e-3) |
| Batch size | 64 |
| Epochs | 40 |

---

### 2. Transformer (QubitClassifierTransformer / MF variant)

**File:** [`networks/Transformer.py`](networks/Transformer.py) (renamed from the typo'd
`Transfomer.py`)

A Vision-Transformer (ViT) inspired encoder for **direct classification from raw IQ traces**
across all 32 states simultaneously. This is the base architecture; it is not, by itself,
the paper's **MCMit-T**.

**MCMit-T** is [`networks/TransformerMF.py`](networks/TransformerMF.py)`::QubitClassifierTransformerMF`
— the same encoder augmented with concatenated HERQULES matched-filter/relaxation-matched-filter
(MF/RMF) features at the classification head (10-D, built via `trainers/train_HERQULES.py`'s
`demodulate_all_qubits`/`compute_all_envelopes`/`build_features`). Trained via
`runners/hyper_optimize_transformer_mf.py`; this MF augmentation is what closes the gap from
the plain transformer's ~0.906 F5Q to the paper's 0.911 (Table 3).

#### Architecture Overview

```
Raw IQ trace (batch, 500, 2)
        │
        ▼
 PatchEmbedding          ← split into 50 patches of 10 samples each, project to 128-D
 + [CLS] token prepended ← learnable token at position 0; final representation → classifier
        │
        ▼
 PositionalEncoding      ← fixed sinusoidal PE added to all 51 tokens
        │
        ▼
 TransformerEncoder      ← 4 × (MHSA + FFN + LayerNorm + Dropout)
   num_heads = 8
   FFN hidden = 512 (4 × embedding_dim)
        │
        ▼
 [CLS] token at index 0  ← aggregates global context via attention
        │
        ▼
 LayerNorm → Linear(128→32) ← raw logits over 32 qubit states
```

#### Default Hyper-parameters

| Parameter | Value |
|---|---|
| `patch_size` | 10 (samples) |
| `embedding_dim` | 128 |
| `num_heads` | 8 |
| `num_layers` | 4 |
| `dropout` | 0.1 |
| `num_classes` | 32 |
| Loss | `nn.CrossEntropyLoss` |

---

### 3. MCMit-CNN

**File:** [`networks/CNN.py`](networks/CNN.py)`::CNN` — a lightweight residual 1-D CNN:
strided-convolution temporal downsampling, `ResidualBlock1D` blocks, global average pooling,
and one binary logit per qubit (multi-task, `in_channels=10, m_param=16, num_qubits=5`).

**Preprocessing:** not `helpers/cnn_helpers.py` (an older path, ~0.908 F5Q) but
`runners/_colleague_prep.py` — FFT-based per-qubit frequency/phase calibration plus a
windowed-sinc sparse FIR demodulator, which is what actually reproduces the paper's 0.911
F5Q (Table 3). See the [caveats above](#caveats--read-before-trusting-a-number).

**Training scripts:** `runners/_cnn_length_sweep.py` (per-length retrain, Table 4) and
`runners/_xtalk_cnn.py` (crosstalk + accuracy-vs-N, Tables 5–6).

[`networks/HybridCNN.py`](networks/HybridCNN.py) is a separate, earlier CNN design kept for
reference — it is *not* MCMit-CNN.

---

### 4. KLiNQ — Knowledge Distillation Pipeline

KLiNQ (**K**nowledge-**Li**ght **N**eural-network **Q**ubit-readout) is a two-stage
distillation framework designed to produce student models small enough for FPGA deployment.

#### Stage 1 — Teacher Training

**File:** [`networks/SingleQubitFNN.py`](networks/SingleQubitFNN.py)

```
Input (2×T)
  └─► Linear(input, max(T, 500)) → BN → ReLU → Dropout(0.5)
  └─► Linear(h1, h1//2)          → BN → ReLU → Dropout(0.5)
  └─► Linear(h2, h2//2)          → BN → ReLU → Dropout(0.5)
  └─► Linear(h3, output)
```

> **Training:** 300 epochs, Adam lr=1e-4, batch=1024, EarlyStopping patience=15.

#### Intermediate Teacher (KLiNQTeacherModel)

**File:** [`networks/KLiNQ_TeacherModel.py`](networks/KLiNQ_TeacherModel.py)

```
Input (2×T) → Linear(→64) → BN → ReLU → Dropout(0.3)
            → Linear(64→32) → BN → ReLU → Dropout(0.3)
            → Linear(32→output)
```

#### Stage 2 — Student Training (KLiNQStudentModel)

**File:** [`networks/KLiNQ_StudentModel.py`](networks/KLiNQ_StudentModel.py)

A tiny model (~250 parameters) trained via knowledge distillation.  Student input is:

| Feature group | Computation | Dimension |
|---|---|---|
| Full flattened IQ trace | `flatten_iq_dimensions(trace[:500, :])` | 1000 |
| Time-averaged IQ | Bin-average to `target_length` bins | 2 × `target_length` |
| Matched-Filter scalar | `I·MF_I + Q·MF_Q` | 1 |

```
Input (input_size) → Linear(→16) → BN → ReLU
                   → Linear(16→8) → BN → ReLU
                   → Linear(8→1)   (raw logit)
```

**Knowledge Distillation Loss:**

```
L = α × L_soft + (1 − α) × L_hard
L_soft = KL( softmax(student/T) || softmax(teacher/T) )
L_hard = BCEWithLogitsLoss(student, true_labels)
```

---

## Repository Structure

```
qubit_state_discrimination/
│
├── HERQULES.py                   ← Full HERQULES training/evaluation pipeline (monolith)
├── matched_filter.py             ← Matched-filter computation utilities
├── train_arxiv_model.py          ← STALE/broken, see caveats — use runners/hyper_optimize.py
├── test.py                       ← Evaluation / inference script
├── fetch_results.py              ← Aggregates optimization_reports/*.csv
├── plot_fidelities.py            ← Plots the master_fidelity.csv schema
├── requirements.txt
│
├── data/
│   ├── five_qubit_data/          ← Place raw HDF5 dataset files here (placeholder only)
│   └── single_qubit_data/        ← Per-qubit datasets (generated by notebooks)
│
├── helpers/
│   ├── config.py                 ← All hyper-parameter and path configuration
│   ├── data_loader.py            ← QubitData class: HDF5 loading + preprocessing
│   ├── data_utils.py             ← Low-level data utilities (normalisation, MF, etc.)
│   ├── nn_utils.py               ← Loss/optimizer setup, DataLoader creation
│   └── cnn_helpers.py            ← Older CNN preprocessing path (NOT used by Tables 3-6)
│
├── networks/
│   ├── __init__.py               ← Package-level exports for all architectures
│   ├── Qubic.py                  ← QubiCML: Arxiv240618807FNN (renamed from Arxiv240618807FNN.py)
│   ├── HERQULES.py               ← Net / Net_rmf (HERQULES's own network classes)
│   ├── HERQULESPlus.py           ← Residual-MLP extension over HERQULES LDA features (unused by Tables 3-6)
│   ├── Transformer.py            ← ViT-style Transformer encoder (renamed from Transfomer.py)
│   ├── TransformerMF.py          ← MCMit-T: Transformer + HERQULES MF/RMF features
│   ├── CNN.py                    ← MCMit-CNN: residual 1-D CNN
│   ├── HybridCNN.py              ← Earlier, non-canonical CNN design (reference only)
│   ├── SingleQubitFNN.py         ← Parametric FNN + SingleQubitFNN_Baseline (Table 3 "Baseline FNN")
│   ├── SingleQubitFNN_StudentModel.py  ← Intermediate KLiNQ student
│   ├── KLiNQ_TeacherModel.py     ← Compact FNN teacher
│   └── KLiNQ_StudentModel.py     ← Tiny student for FPGA deployment
│
├── trainers/
│   ├── HERQULES_original.py      ← Canonical HERQULES pipeline (Tables 3-6)
│   ├── train_HERQULES.py         ← Alternate "faithful replication" HERQULES (~0.925, not Table 3)
│   ├── train_HERQULESPlus.py
│   └── ...                       ← KD and SingleQubitFNN training logic (KLiNQ)
│
└── runners/                      ← Executable training/eval scripts
    ├── hyper_optimize.py                 ← Table 3: Baseline FNN + QubiCML
    ├── hyper_optimize_transformer_mf.py  ← Table 3: MCMit-T
    ├── _colleague_prep.py                ← MCMit-CNN preprocessing (imported, not run directly)
    ├── _cnn_length_sweep.py              ← Tables 3-4: MCMit-CNN
    ├── _xtalk_cnn.py                     ← Tables 5-6: MCMit-CNN
    ├── _herqules_truncation_pertrace.py  ← Table 4: HERQULES
    ├── _herqules_truncation_eval.py      ← Table 4 alt (subsampled data)
    ├── _herqules_length_both.py          ← Table 3 alt: full per-qubit HERQULES CSV row
    ├── _xtalk_herqules_deploy.py         ← Tables 5-6: HERQULES
    ├── train_SingleQubitFNN.py
    ├── train_KD_with_SingleQubitFNN.py
    └── train_KD_with_KLinQ_TeacherStudent.py
```

---

## Data Pipeline

### `helpers/data_utils.py`

Low-level, stateless utility functions operating on NumPy arrays.

| Function | Description |
|---|---|
| `hdf5_data_load` | Load `X` and `y` from an HDF5 file. |
| `custom_hdf5_data_loader` | Memory-efficient partial HDF5 load. |
| `QubitTraceDataset` | `torch.utils.data.Dataset` wrapper for NumPy arrays. |
| `reduce_trace_duration` | Truncate `(N, T, 2)` → `(N, T', 2)`. |
| `flatten_iq_dimensions` | Reshape `(N, T, 2)` → `(N, 2T)` for FNN models. |
| `stratified_split` | Class-balanced train/val split. |
| `normalize_data` | z-score normalisation (creates new arrays). |
| `normalize_data_inplace` | z-score normalisation in-place. |
| `normalize_data_forb` | Frobenius-norm division. |
| `normalize_data_std_p2` | z-score with std rounded to nearest power of 2 (FPGA-friendly). |
| `apply_mf_rmf` | Compute matched-filter scalar: `output = I·MF_I + Q·MF_Q`. |
| `compute_normalization_params` | Compute `{n, mu}` for fixed-point normalisation. |
| `apply_normalization` | Apply fixed-point-friendly normalisation. |

### `helpers/data_loader.py`

High-level `QubitData` class orchestrating the full preprocessing pipeline.

| Method | Pipeline |
|---|---|
| `load_data()` | Load raw HDF5 → `(X_train, y_train, X_test, y_test)`. |
| `transform(...)` | Truncate → flatten → normalise → split. |
| `load_transform()` | **Standard pipeline** for FNN and Transformer. |
| `load_transform_KLiNQ_KD(target_length)` | **KLiNQ pipeline**: full trace + averaged trace + MF scalar. |
| `average_trace_data_fixed_length(data, n)` | Bin-average traces to `n` time bins. |

**Normalisation strategies** (selected via `data_config['normalize']`):

| Key | Method |
|---|---|
| `'mean/std'` | z-score normalisation |
| `'forb'` | Frobenius-norm division |
| `'forb_s'` | Frobenius subtraction variant |
| `'forb-weighted'` | Frobenius / 4× |
| `'mean/p2std'` | z-score with power-of-2 std (FPGA-friendly) |
| `'no-norm'` | No normalisation |

---

## Usage Instructions

### 1. Place raw data

Put the original HDF5 dataset files in `data/five_qubit_data/`:
- `DRaw_C_Tr_v0-001`  (training)
- `DRaw_C_Te_v0-002`  (testing)

### 2. Demodulate the multiplexed traces

Run the demodulation step to produce per-qubit HDF5 files:

```python
from HERQULES import demodulate_multiplexed_traces
import numpy as np

demodulate_multiplexed_traces(
    iq_traces=all_data,
    qubit_frequencies=freq_readout,
    sampling_rate=500e6
)
# Produces: demodulated_q1_.h5 ... demodulated_q5_.h5
```

### 3. Run the HERQULES training pipeline

```python
from HERQULES import train

# Full pipeline: pre-classifier + MF + RMF + neural network
acc_per_qubit = train(run_semi_sup=True, run_rmf=True)
```

This will:
1. Fit the `preclassifier` and save `preclassifier_state.pkl`.
2. Compute and save matched-filter envelopes.
3. Fit the `relaxation_mf_classifier` and save `rmf.pkl`.
4. Train `Net_rmf` and save `checkpoints/mf_nn/best_epoch.pth`.
5. Print overall and per-qubit test accuracy.

### 4. Evaluate a saved model

```python
from HERQULES import test

overall_acc, per_qubit_acc = test()
```

### 5. Run the baseline or neural network models

**QubiCML / Baseline FNN (paper Table 3):**
```bash
python runners/hyper_optimize.py
```
(`train_arxiv_model.py` at the repo root imports a stale module path and is broken as
committed — use `hyper_optimize.py` instead; see the
[caveats](#caveats--read-before-trusting-a-number) at the top of this README.)

**KLiNQ teacher training (per qubit):**
```bash
python runners/train_SingleQubitFNN.py
```

**KLiNQ knowledge distillation:**
```bash
python runners/train_KD_with_KLinQ_TeacherStudent.py
```

### 6. Generate per-qubit datasets for KLiNQ

```bash
jupyter notebook data/single_qubit_dataset_creator.ipynb
jupyter notebook data/multiplexed_traces_mf_rmf_save.ipynb
```

---

## Development Story of KLiNQ

1. **Data preparation** — multiplexed 5-qubit IQ traces split into 5 individual single-qubit
   datasets using the `single_qubit_dataset_creator` notebook.

2. **Teacher training** — `SingleQubitFNN` models (e.g. layers `[1000, 500, 250]`) trained
   independently per qubit.

3. **Architecture search** — many FNN, CNN, and recurrent architectures tested;
   `SingleQubitFNN` consistently outperformed alternatives.

4. **Stage-1 distillation** — trained `SingleQubitFNN` teachers distilled into smaller
   `SingleQubitStudentModel` networks.  Many student models *outperformed* their teachers
   — a well-known regularisation effect of knowledge distillation.

5. **Best student as new teacher** — top-performing stage-1 students re-used as teachers
   for stage 2.

6. **Stage-2 distillation (KLiNQ)** — student takes compact feature vector (averaged IQ +
   MF scalar).  Tiny architectures `[31, 16, 8, 1]` and `[201, 16, 8, 1]` explored,
   targeting FPGA resource budgets.

---

## Disclaimer & Research Context

> **NOTE:** This repository is not in its final production-ready shape.  The codebase has
> not been fully cleaned or polished due to time constraints.  The raw dataset is not
> uploaded to GitHub for space and policy reasons.

> **REMARK:** This repository does **not** contain the codebase for all experiments
> conducted during KLiNQ development.