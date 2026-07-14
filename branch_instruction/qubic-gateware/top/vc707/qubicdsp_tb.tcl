set target "qubicdsp"
set part "xc7vx485tffg1761-2"
set outputdir ./vivado_project_sim
set depd ${target}_tb.d
set tend [lindex $argv 0]
source submodules/tools/proj.tcl
source submodules/tools/depsrc.tcl
source submodules/tools/sim.tcl
proj ${target} ${part} ${outputdir} ${depd}
sim ${target} ${tend}
