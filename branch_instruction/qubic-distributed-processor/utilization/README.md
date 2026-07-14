# Utilization

To compare the resource utilization between the unmodified distributed processor and our version including the branch_reduce instruction, we put the sources in hdl/ and sim_modules/toplevel_sim.sv into a standalone Vivado project. We ran the implementation step and got the resource utilization reports for both the `main` branch (unmod) and the `branch-reduce` branch (mod).
