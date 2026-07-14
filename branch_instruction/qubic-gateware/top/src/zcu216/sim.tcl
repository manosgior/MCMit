#open_project /tmp/pynqtop/zcu216/rfsoctest/boards/zcu216/vivado_project/psbd/psbd.xpr
add_files -fileset sim_1 -norecurse psbd_wrapper_tb.sv
add_files -fileset sim_1 -norecurse ../../submodules/board-support/zcu216/fpga_if.sv
add_files -fileset sim_1 -norecurse ../../submodules/board-support/zcu216/board.sv
add_files -fileset sim_1 -norecurse ../../submodules/board-support/zcu216/hwif.sv
set_property top psbd_wrapper_tb [get_filesets sim_1]

#launch_simulation


