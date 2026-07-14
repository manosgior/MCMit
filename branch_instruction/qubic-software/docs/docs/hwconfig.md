ElementConfig implementation for current QubiC gateware running on the RFSoC (ZCU216) platform. Responsible for converting phase/frequencies/amplitudes from natural units into words/buffers that can be loaded into FPGA memory. Instantiated during [assembly](toolchain.md/#qubic.toolchain.run_assemble_stage) based on provided channel config.

::: qubic.rfsoc.hwconfig
