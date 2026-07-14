The `qubic` module contains the core software tools for compiling and running QubiC programs, as well as conducting simple data analysis such as state classification. 

- **circuit compilation**: [qubic.toolchain](toolchain.md) wraps the core qubic compiler tools from the [distributed processor](https://gitlab.com/LBL-QubiC/distributed_processor/-/tree/master/python) repo. Compilation + assembling lowers the QubiC [high-level program representation](https://lbl-qubic.gitlab.io/distributed_processor) to [distributed processor assembly](https://gitlab.com/LBL-QubiC/distributed_processor/-/wikis/Instruction-Set) code.
- **infrastructure for running circuits** (see the [Getting Started Guide](https://gitlab.com/LBL-QubiC/software/-/wikis/Getting-Started-with-QubiC-2.0-on-the-ZCU216) for how to set this up):
    - From a user standpoint, the **lowest-level** interface you will interact with is defined by the [AbstractCircuitRunner](abstract_runner.md) class, which allows you to run **compiled QubiC programs** and receive the resulting raw data from hardware (in the form of **ADC traces** or **integrated I/Q data**). Depending on the implementation, the AbstractCircuitRunner can be accessed on the ZCU216 ([CircuitRunner](run.md)) or on a client machine ([CircuitRunnerClient](rpc_client.md/#qubic.rpc_client.CircuitRunnerClient)). 
    - The [JobManager](job_manager.md/#qubic.job_manager.JobManager) is a **higher-level** interface, which wraps **compilation, execution**, and (optionally) **state classification**
    - **Job servers** can be configured using [qubic.soc_rpc_server](soc_rpc_server.md) and/or [qubic.job_rpc_server](job_rpc_server.md) to allow compiled circuits to be submitted from a remote machine
- **analysis tools**: [qubic.state_discrimination](state_disc.md) for **qudit state classification** using a Gaussian-mixture model (GMM).
- interfaces to **[TrueQ](https://trueq.quantumbenchmark.com)** ([qubic.trueq]()) and **[PyGSTi](https://www.pygsti.info)** ([qubic.pygsti]())

## Helpful Links for Getting Started

- [Getting Started Guide](https://gitlab.com/LBL-QubiC/software/-/wikis/Getting-Started-with-QubiC-2.0-on-the-ZCU216) for configuring the ZCU216 to run QubiC
- [Understanding Channel Configuration](https://gitlab.com/LBL-QubiC/software/-/wikis/Understanding-Channel-Configuration) for the channel layout of the current gateware, and the gateware -> hardware channel mapping.
- [demo notebooks](https://gitlab.com/LBL-QubiC/software/-/tree/master/examples?ref_type=heads) for writing and running simple qubic programs
- more advanced examples can be found in the [repo](https://gitlab.com/LBL-QubiC/qce23tutorial) for our [IEEE Quantum Week 2023 Tutorial](https://sites.google.com/lbl.gov/qubicqce23?usp=sharing)
- language references: 
    - [distributed processor instruction set](https://gitlab.com/LBL-QubiC/distributed_processor/-/wikis/Instruction-Set)
    - [high-level program representation](https://lbl-qubic.gitlab.io/distributed_processor)
