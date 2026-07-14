set target "qubic"
set part "xc7vx485tffg1761-2"
set outputdir ./vivado_project
set depd ${target}.d
source submodules/tools/proj.tcl
source submodules/tools/depsrc.tcl
source submodules/tools/synimpbit.tcl
proj ${target} ${part} ${outputdir} ${depd}
synimpbit $target 11
