set target "plps"
set part "xczu49dr-ffvf1760-2-e"
set outputdir ./vivado_project/plps
set depd ${target}.d
set submodulepath ../../../submodules
source ${submodulepath}/tools/proj.tcl
source ${submodulepath}/tools/depsrc.tcl
source ${submodulepath}/tools/synimpbit.tcl
proj ${target} ${part} ${outputdir} ${depd}
source ../../src/plpsip.tcl
#ipx::package_project  -root_dir ./vivado_project/plip -set_current false
#-vendor user.org -library user -taxonomy /UserIP -import_files 
#synimpbit $target 11
#close_project
