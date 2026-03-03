# qubit_readout_klinq

This repository implements deep-learning classifiers for **qubit state discrimination** in superconducting quantum computers.  It covers three distinct neural network approaches, from a simple FNN that replicates a published paper, to a full Vision-Transformer, to KLiNQ — a knowledge-distillation pipeline designed to produce tiny models suitable for FPGA deployment.

---

## Table of Contents

1. [Physical Background](#physical-background)
2. [Model Architectures](#model-architectures)
   - [arXiv:2406.18807 FNN](#1-arxiv240618807-fnn)
   - [Transformer (QubitClassifierTransformer)](#2-transformer-qubitclassifiertransformer)
   - [KLiNQ — Knowledge Distillation Pipeline](#3-klinq--knowledge-distillation-pipeline)
3. [Repository Structure](#repository-structure)
4. [Data Pipeline](#data-pipeline)
   - [`helpers/data_utils.py`](#helpersdatautilspy)
   - [`helpers/data_loader.py`](#helpersdataloaderpy)
5. [Usage Instructions](#usage-instructions)
6. [Development Story of KLiNQ](#development-story-of-klinq)
7. [Disclaimer & Research Context](#disclaimer--research-context)

---

## Physical Background

In dispersive qubit readout a microwave readout tone is sent through a resonator coupled to the qubit.  The qubit state (|0⟩ or |1⟩) shifts the resonator frequency, producing a characteristic phase and amplitude response in the reflected or transmitted signal.  The returned signal is **downconverted** to an intermediate frequency (IF) and digitised by an ADC, yielding two quadrature components:

- **I** (in-phase)
- **Q** (quadrature)

For **multiplexed readout** of 5 qubits, all five resonators are probed simultaneously at different IF frequencies in a shared bandwidth.  The dataset files contain raw IQ traces of shape `(N_shots, trace_length, 2)` and integer labels in `[0, 31]` (one bit per qubit).

---

## Model Architectures

### 1. arXiv:2406.18807 FNN

**File:** [`networks/Arxiv240618807FNN.py`](networks/Arxiv240618807FNN.py)  
**Training script:** [`train_arxiv_model.py`](train_arxiv_model.py)

A reproduction of the lightweight FNN described in [arXiv:2406.18807](https://arxiv.org/abs/2406.18807).

#### Architecture

```
Input (2)  ──► Linear(2→8) ──► ReLU ──► Linear(8→4) ──► ReLU ──► Linear(4→1) ──► Sigmoid
```

| Layer        | Size | Activation |
|--------------|------|------------|
| Input        | 2    | —          |
| Hidden 1     | 8    | ReLU       |
| Hidden 2     | 4    | ReLU       |
| Output       | 1    | Sigmoid    |

#### Input Processing (Multiplexed Case)

The raw 5-qubit multiplexed trace is **demodulated** per qubit before being fed to the network:

1. **Frequency Demodulation** — for each of the 5 qubit IF frequencies, multiply the trace by a local oscillator (LO) complex exponential `e^{j2πf_IF t}` and integrate (average) over the readout window.  This isolates the IQ contribution from each qubit's resonator.
2. **Min-max normalisation** — each (I, Q) column is scaled to [0, 1].
3. **Per-qubit labelling** — integer state label is bit-shifted: `y_q = (y >> q) & 1`.
4. **One model per qubit** — 5 separate instances of `Arxiv240618807FNN` are trained independently.

#### Training Configuration (paper §6.2)

| Parameter | Value |
|-----------|-------|
| Total samples | 300,000 (160k train + 140k test combined, then split 60/40) |
| Loss | Binary Cross-Entropy (`nn.BCELoss`) |
| Optimiser | Adam (lr = 1e-3) |
| Batch size | 64 |
| Epochs | 40 |

---

### 2. Transformer (QubitClassifierTransformer)

**File:** [`networks/Transfomer.py`](networks/Transfomer.py)

A Vision-Transformer (ViT) inspired encoder for **direct classification from raw IQ traces** across all 32 states simultaneously (no per-qubit demodulation required).

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

#### Component Details

| Component | Role |
|-----------|------|
| `PatchEmbedding` | Divides trace into 50 non-overlapping windows of 10 samples; each window's 20 values (10×I + 10×Q) are projected to 128-D via `nn.Linear`. A learnable `[CLS]` token is prepended. |
| `PositionalEncoding` | Adds fixed sinusoidal encodings (Vaswani et al. 2017) so the model knows patch order. |
| `TransformerEncoderLayer` | Standard PyTorch layer: Multi-Head Self-Attention (residual) → LayerNorm → 2-layer FFN (residual) → LayerNorm → Dropout. `batch_first=True`. |
| Classification head | `LayerNorm → Linear(128, num_classes)`. Applied to the `[CLS]` token only. |

#### Default Hyper-parameters

| Parameter | Value |
|-----------|-------|
| `patch_size` | 10 (samples) |
| `embedding_dim` | 128 |
| `num_heads` | 8 |
| `num_layers` | 4 |
| `dropout` | 0.1 |
| `num_classes` | 32 |
| Loss | `nn.CrossEntropyLoss` |

---

### 3. KLiNQ — Knowledge Distillation Pipeline

KLiNQ (**K**nowledge-**Li**ght **N**eural-network **Q**ubit-readout) is a two-stage distillation framework designed to produce student models small enough to be deployed on an FPGA alongside the qubit readout hardware.

#### Stage 1 — Teacher Training

**File:** [`networks/SingleQubitFNN.py`](networks/SingleQubitFNN.py)

A deep FNN with three adaptive hidden layers is trained on the full flattened IQ trace.

```
Input (2×T)
  └─► Linear(input, max(T, 500)) → BN → ReLU → Dropout(0.5)
  └─► Linear(h1, h1//2)          → BN → ReLU → Dropout(0.5)
  └─► Linear(h2, h2//2)          → BN → ReLU → Dropout(0.5)
  └─► Linear(h3, output)
```

Hidden layer widths are derived from `input_size` at construction time:  
`h1 = max(input_size, 500)`, `h2 = h1 // 2`, `h3 = h2 // 2`.

> **Training hyper-parameters:** 300 epochs, Adam lr=1e-4, batch=1024, EarlyStopping patience=15.

#### Intermediate Teacher (KLiNQTeacherModel)

**File:** [`networks/KLiNQ_TeacherModel.py`](networks/KLiNQ_TeacherModel.py)

A smaller FNN explored as an alternative teacher:

```
Input (2×T) → Linear(→64) → BN → ReLU → Dropout(0.3)
            → Linear(64→32) → BN → ReLU → Dropout(0.3)
            → Linear(32→output)
```

#### Stage 2 — Student Training (KLiNQStudentModel)

**File:** [`networks/KLiNQ_StudentModel.py`](networks/KLiNQ_StudentModel.py)

A tiny model (~250 parameters) trained via knowledge distillation from the best-performing teacher.

**Student input** is a carefully designed composite feature vector (not the raw trace):

| Feature group | How it is computed | Dimension |
|---|---|---|
| Full flattened IQ trace | `flatten_iq_dimensions(trace[:500, :])` | 1000 |
| Time-averaged IQ | Trace divided into `target_length` bins; bin-average computed; flattened | 2 × target_length |
| Matched-Filter scalar | Dot product of trace with MF pulse template (I and Q) | 1 |

These three groups are **column-stacked** (`np.column_stack`) and passed to the student.  Typical input sizes from the paper: **31** (short traces) or **201** (medium traces).

```
Input (input_size) → Linear(→16) → BN → ReLU
                   → Linear(16→8) → BN → ReLU
                   → Linear(8→1)   (raw logit)
```

**Knowledge Distillation Loss:**

```
L = α × L_soft + (1 − α) × L_hard

L_soft = KL( softmax(student_logits/T) || softmax(teacher_logits/T) )
L_hard = BCEWithLogitsLoss(student_logits, true_labels)
```

Multiple (T, α) configurations are explored (see `helpers/config.py`).

---

## Repository Structure

```
qubit_readout_klinq/
├── data/
│   ├── five_qubit_data/          ← Place raw HDF5 files here
│   └── single_qubit_data/        ← Per-qubit datasets generated by notebook
│
├── helpers/
│   ├── config.py                 ← All hyper-parameter and path configuration
│   ├── data_loader.py            ← QubitData class: HDF5 loading + preprocessing
│   ├── data_utils.py             ← Low-level data utilities (normalisation, MF, etc.)
│   └── nn_utils.py               ← Loss/optimizer setup, DataLoader creation
│
├── networks/
│   ├── Arxiv240618807FNN.py      ← 2-hidden-layer FNN (arXiv:2406.18807)
│   ├── Transfomer.py             ← ViT-style Transformer encoder
│   ├── SingleQubitFNN.py         ← Large adaptive FNN (KLiNQ teacher)
│   ├── SingleQubitFNN_StudentModel.py  ← Intermediate student
│   ├── KLiNQ_TeacherModel.py     ← Compact FNN teacher
│   └── KLiNQ_StudentModel.py     ← Tiny student for FPGA deployment
│
├── trainers/                     ← KD and standard training logic
├── runners/                      ← Executable training scripts
│   ├── train_SingleQubitFNN.py
│   ├── train_KD_with_SingleQubitFNN.py
│   └── train_KD_with_KLinQ_TeacherStudent.py
│
├── artifacts/
│   ├── original_models/          ← Trained teacher checkpoints
│   └── distilled_models/         ← Trained student checkpoints
│
├── train_arxiv_model.py          ← Training script for arXiv FNN (5-qubit)
└── test.py                       ← Evaluation / inference script
```

---

## Data Pipeline

### `helpers/data_utils.py`

Low-level, stateless utility functions.  All functions operate on NumPy arrays.

| Function | Description |
|---|---|
| `hdf5_data_load` | Load `X` and `y` from an HDF5 file (`'X_train'`/`'y_train'` or `'X_test'`/`'y_test'` key). |
| `custom_hdf5_data_loader` | Load only a fraction of an HDF5 file without reading the entire file (memory-efficient). |
| `QubitTraceDataset` | `torch.utils.data.Dataset` wrapper that converts NumPy arrays to typed tensors. |
| `reduce_trace_duration` | Truncate `(N, T, 2)` traces to `(N, reduction_size, 2)` by slicing the time axis. |
| `flatten_iq_dimensions` | Reshape `(N, T, 2)` → `(N, 2T)` for FNN models. |
| `stratified_split` | Class-balanced train/val split (equal samples per class). |
| `normalize_data` | z-score normalisation using training-set statistics (creates new arrays). |
| `normalize_data_inplace` | Same as above but modifies arrays in-place (memory-efficient). |
| `normalize_data_forb` | Divide by the Frobenius norm of the training matrix. |
| `normalize_data_forb_weighted` | Same but divides by `4 × Frobenius norm`. |
| `normalize_data_forb_subtraction` | Per-sample norm subtraction + std scaling. |
| `normalize_data_std_p2` | z-score with std rounded to nearest power of 2 (FPGA-friendly). |
| `nearest_power_of_2` | Round array values to nearest power of 2 (via log2 rounding). |
| `apply_mf_rmf` | Compute the matched-filter (or RMF) scalar output: `output = I·MF_I + Q·MF_Q`. |
| `compute_normalization_params` | Compute `{n, mu}` per IQ component for fixed-point normalisation. |
| `apply_normalization` | Apply fixed-point-friendly normalisation using pre-computed parameters. |

### `helpers/data_loader.py`

High-level `QubitData` class that orchestrates the full preprocessing pipeline.

| Method | Pipeline |
|---|---|
| `load_data()` | Load raw HDF5 arrays → `(X_train, y_train, X_test, y_test)`. |
| `transform(...)` | Truncate → flatten → normalise → stratified split.  Accepts pre-loaded arrays. |
| `load_transform()` | **Standard pipeline** (for FNN and Transformer).  Calls `load_data()` then `transform`. |
| `load_transform_KLiNQ_KD(target_length)` | **KLiNQ pipeline**.  Builds composite feature vector: full trace + averaged trace + MF scalar. |
| `average_trace_data_fixed_length(data, target_length)` | Bin-average a batch of traces from `trace_length` bins to `target_length` bins. |

**Normalisation strategy** is selected via `data_config['normalize']`:

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

### 2. Generate per-qubit datasets and MF/RMF features

Run the two Jupyter notebooks (in order):

```bash
jupyter notebook data/single_qubit_dataset_creator.ipynb
jupyter notebook data/multiplexed_traces_mf_rmf_save.ipynb
```

This produces 10 per-qubit HDF5 files and MF/RMF envelope pickle files in `data/single_qubit_data/`.

### 3. Update the project root path

Edit `helpers/config.py`:
```python
root_dir = Path("/absolute/path/to/qubit_readout_klinq")
```

### 4. Run training scripts

**arXiv FNN (5-qubit multiplexed):**
```bash
python train_arxiv_model.py
```

**SingleQubitFNN teacher (per qubit):**
```bash
python runners/train_SingleQubitFNN.py
```

**Knowledge distillation — SingleQubitFNN teacher:**
```bash
python runners/train_KD_with_SingleQubitFNN.py
```

**Knowledge distillation — KLiNQ teacher→student:**
```bash
python runners/train_KD_with_KLinQ_TeacherStudent.py
```

---

## Development Story of KLiNQ

1. **Data preparation** — multiplexed 5-qubit IQ traces were split into 5 individual single-qubit datasets using the `single_qubit_dataset_creator` notebook.

2. **Teacher training** — `SingleQubitFNN` models (layers e.g. `[1000, 500, 250]`) were trained for each qubit independently.

3. **Architecture search** — many FNN, CNN, and recurrent architectures were tested, but `SingleQubitFNN` consistently outperformed alternatives on this dataset.

4. **Stage-1 distillation** — the trained `SingleQubitFNN` teachers distilled into smaller `SingleQubitStudentModel` networks.  Interestingly, many student models *outperformed* their teachers — a well-known effect of knowledge distillation acting as a regulariser.

5. **Best student as new teacher** — the best-performing stage-1 students (e.g. `[1000, 64, 32]`) were re-used as teachers for stage 2.

6. **Stage-2 distillation (KLiNQ)** — the KLiNQ student takes a compact feature vector (averaged IQ + MF scalar) instead of the full trace.  Tiny architectures such as `[31, 16, 8, 1]` and `[201, 16, 8, 1]` were explored, targeting FPGA resource budgets.

---

## Disclaimer & Research Context

> **NOTE:** This repository is not in its final production-ready shape.  The codebase has not been fully cleaned or polished due to time constraints.  The raw dataset is not uploaded to GitHub for space and policy reasons.  Trained model artifacts are preserved in `artifacts/`.

> **REMARK:** This repository does **not** contain the codebase for all experiments conducted during KLiNQ development.