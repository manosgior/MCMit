# branch_instruction

This directory holds the FPGA controller implementation behind the paper's
**`branch_reduce_fproc`** instruction (§5, "Enabling Scalable Classical
Feedback") — the scalable, constant-latency multi-qubit branch instruction
described in §5.2 and Fig. 6, benchmarked in Table 2.

## What's here

MCMit's controller is built on top of the open-source
[QubiC 2.0](https://gitlab.com/LBL-QubiC) framework (Lawrence Berkeley
National Lab). Per §8.1: *"We implement the MCMit
controller on top of the open-source Qubic 2.0 framework by modifying the
Qubic software (Qubic IR, compiler, assembler, and command generation), the
Qubic distributed processor, and the Qubic gateware."* Those are exactly
the three subdirectories here — each is a fork of the corresponding
upstream LBL-QubiC repo, with MCMit's modifications on the `branch-reduce`
branch:

| Directory | Fork of (upstream) | Branch |
|---|---|---|
| [`qubic-distributed-processor/`](qubic-distributed-processor/) | [github.com/TUM-DSE/qubic-distributed-processor](https://github.com/TUM-DSE/qubic-distributed-processor/tree/branch-reduce) (upstream: [gitlab.com/LBL-QubiC/distributed_processor](https://gitlab.com/LBL-QubiC)) | `branch-reduce` |
| [`qubic-gateware/`](qubic-gateware/) | [github.com/TUM-DSE/qubic-gateware](https://github.com/TUM-DSE/qubic-gateware/tree/branch-reduce) (upstream: [gitlab.com/LBL-QubiC/gateware](https://gitlab.com/LBL-QubiC)) | `branch-reduce` |
| [`qubic-software/`](qubic-software/) | [github.com/TUM-DSE/qubic-software](https://github.com/TUM-DSE/qubic-software/tree/branch-reduce) (upstream: [gitlab.com/LBL-QubiC/software](https://gitlab.com/LBL-QubiC)) | `branch-reduce` |

Each subdirectory is a **snapshot** of that branch's current state (no git
history carried over — see [Provenance](#provenance) below for how to get
that from the source).

## The three components, and how they map to §5

### `qubic-distributed-processor/` — the ALU, control unit, and command format

The core of Fig. 6's modifications. Contains both the FPGA HDL (`hdl/`,
Verilog/SystemVerilog) for the distributed processor core and its Python
compiler/assembler stack (`python/`, `distproc`: compiles gate-level
programs, including measurement-based control flow, to distributed
processor machine code).

This is where the paper's §5.2 changes live:
- **ALU**: widened opcode supporting `and`, `or`, `xor`, `maj` (majority
  vote) over up to 32 qubit measurement inputs with a 32-bit qubit-select
  mask.
- **Command buffer**: extended from 128 to 160 bits (5×32-bit bus) to carry
  the wider ALU opcode.
- **Function processor**: modified so the distributed processor core
  receives the *whole* 32-bit measurement register (all qubits) instead of
  just the single least-significant bit, for `branch_reduce_fproc`.

### `qubic-gateware/` — the full FPGA build

The complete QubiC 2.0 gateware build (Vivado 2022.1, ZCU216 RFSoC target)
that instantiates the distributed processor core above as part of the
whole control system (readout converter, readout/qubit drive, DAC/ADC
interfaces — Fig. 6). Has cocotb testbenches exercising the new
branch/branch_reduce instructions.

**Not vendored here**: this repo's own build depends on four further git
submodules from the original LBL-QubiC project
(`board-support`, `common-hdl`, `fpga-family`, `tools` — all unmodified by
MCMit) plus a submodule reference back to `qubic-distributed-processor`
itself (which is already vendored as a sibling directory above — same
content, not duplicated here). To actually build the gateware, clone the
[source repo](https://github.com/TUM-DSE/qubic-gateware/tree/branch-reduce)
directly and run `git submodule update --init --recursive`.

### `qubic-software/` — the host-side control software

The Python software stack that runs on the host (Qubic IR, compiler,
assembler, command generation, calibration, experiment management — the
"Qubic software" named in §8.1). Depends on `distproc` from
`qubic-distributed-processor/python`.

## Provenance

These are **snapshots**, not vendored with git history, to avoid pulling
the full upstream LBL-QubiC lineage (hundreds to thousands of commits
spanning years of the original project) into MCMit's own history. For the
full development history, see each repo's `branch-reduce` branch directly
on GitHub (linked in the table above), or diff it against `main` there.

## What wasn't touched

Nothing else in MCMit changed as part of this import — this is purely
adding these three snapshots under `branch_instruction/`.
