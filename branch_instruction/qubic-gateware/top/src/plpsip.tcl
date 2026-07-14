proc portpin {busname portname physical_name} {
ipx::add_port_map ${portname} [ipx::get_bus_interfaces ${busname}  -of_objects [ipx::current_core]]
set_property physical_name ${physical_name} [ipx::get_port_maps ${portname} -of_objects [ipx::get_bus_interfaces ${busname} -of_objects [ipx::current_core]]]
}
proc lbbus {lbname} {
ipx::add_bus_interface ${lbname} [ipx::current_core]
set_property abstraction_type_vlnv user:user:iflocalbus_rtl:1.0 [ipx::get_bus_interfaces ${lbname} -of_objects [ipx::current_core]]
set_property bus_type_vlnv user:user:iflocalbus:1.0 [ipx::get_bus_interfaces ${lbname} -of_objects [ipx::current_core]]
set pin {rdata raddr rden rdenlast wdata waddr wren rvalid rvalidlast clk aresetn}
foreach p $pin {
	portpin ${lbname} $p ${lbname}_$p
}
}
proc brambus {bramname} {
	ipx::add_bus_interface ${bramname} [ipx::current_core]
	set_property abstraction_type_vlnv xilinx.com:interface:bram_rtl:1.0 [ipx::get_bus_interfaces ${bramname} -of_objects [ipx::current_core]]
	set_property bus_type_vlnv xilinx.com:interface:bram:1.0 [ipx::get_bus_interfaces ${bramname} -of_objects [ipx::current_core]]
	set_property interface_mode master [ipx::get_bus_interfaces ${bramname} -of_objects [ipx::current_core]]
	bram_map ${bramname} [string toupper $bramname]
}
proc clkbus {clkname} {
	ipx::add_bus_interface ${clkname} [ipx::current_core]
	set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 [ipx::get_bus_interfaces ${clkname} -of_objects [ipx::current_core]]
	set_property bus_type_vlnv xilinx.com:signal:clock:1.0 [ipx::get_bus_interfaces ${clkname} -of_objects [ipx::current_core]]
	set_property interface_mode master [ipx::get_bus_interfaces ${clkname} -of_objects [ipx::current_core]]
	ipx::add_bus_parameter FREQ_HZ [ipx::get_bus_interfaces ${clkname} -of_objects [ipx::current_core]]
	ipx::add_port_map CLK [ipx::get_bus_interfaces ${clkname} -of_objects [ipx::current_core]]
	set_property physical_name ${clkname} [ipx::get_port_maps CLK -of_objects [ipx::get_bus_interfaces ${clkname} -of_objects [ipx::current_core]]]
	ipx::remove_bus_parameter FREQ_HZ [ipx::get_bus_interfaces ${clkname} -of_objects [ipx::current_core]]
}

proc iopin {pin masterdir} {
	ipx::add_bus_abstraction_port ${pin} [ipx::current_busabs]
	if [string equal $masterdir "in"] {
		set_property -dict {default_value {0} master_presence {required} master_direction {in} slave_presence {required} slave_direction {out}} [ipx::get_bus_abstraction_ports ${pin} -of_objects [ipx::current_busabs]]
	} elseif [string equal $masterdir "out"] {
		set slavedir "in"
		set_property -dict {default_value {0} master_presence {required} master_direction {out} slave_presence {required} slave_direction {in}} [ipx::get_bus_abstraction_ports ${pin} -of_objects [ipx::current_busabs]]

	} elseif [string equal $masterdir "inout"] {
		set slavedir "inout"
		set_property -dict {default_value {0} master_presence {required} master_direction {inout} slave_presence {required} slave_direction {inout}} [ipx::get_bus_abstraction_ports ${pin} -of_objects [ipx::current_busabs]]

	} else {
		puts "direction wrong"
	}
}

proc fpgapinmap {pin} {
	ipx::add_port_map ${pin} [ipx::get_bus_interfaces fpga -of_objects [ipx::current_core]]
	set_property physical_name FPGA_${pin} [ipx::get_port_maps ${pin} -of_objects [ipx::get_bus_interfaces fpga -of_objects [ipx::current_core]]]
}
proc fpgamap {pindirfile} {
	set fp [open $pindirfile r]
	set str [read $fp]
	foreach pin [split $str "\n"] {
		fpgapinmap [lindex $pin 0]
	}
}

proc fpgaif {pindirfile} {

	ipx::create_abstraction_definition fpga user fpga_rtl 1.0
	ipx::create_bus_definition fpga user fpga 1.0
	set_property xml_file_name ./vivado_project/plip/fpga_rtl.xml [ipx::current_busabs]
	set_property xml_file_name ./vivado_project/plip/fpga.xml [ipx::current_busdef]
	set_property bus_type_vlnv fpga:user:fpga:1.0 [ipx::current_busabs]

	set fp [open $pindirfile r]
	set str [read $fp]
	foreach pin [split $str "\n"] {
		iopin [lindex $pin 0] [lindex $pin 1]
	}

	ipx::save_bus_definition [ipx::current_busdef]
	set_property bus_type_vlnv fpga:user:fpga:1.0 [ipx::current_busabs]
	ipx::save_abstraction_definition [ipx::current_busabs]
}
proc bram_map {busname inst} {
	set brampins {RST CLK DIN EN DOUT WE ADDR}
	foreach pin $brampins {
		ipx::add_port_map $pin [ipx::get_bus_interfaces $busname -of_objects [ipx::current_core]]
		set_property physical_name ${inst}_[string tolower $pin] [ipx::get_port_maps $pin -of_objects [ipx::get_bus_interfaces $busname -of_objects [ipx::current_core]]]
	}
}


#open_project ./vivado_project/plframe/plframe.xpr
ipx::package_project -root_dir ./vivado_project/plip -vendor user.org -library user -taxonomy /UserIP -import_files -set_current false -force -quiet
ipx::package_project -root_dir ./vivado_project/plip -vendor user.org -library user -taxonomy /UserIP -import_files -set_current false -force
ipx::unload_core ./vivado_project/plip/component.xml
ipx::open_ipxact_file ./vivado_project/plip/component.xml

#ipx::edit_ip_in_project -upgrade true -name plip -directory ./vivado_project/plip ./vivado_project/plip/component.xml
if {0} {
brambus qdrvenv0
brambus qdrvenv1
brambus qdrvenv2
brambus qdrvenv3
brambus qdrvenv4
brambus qdrvenv5
brambus qdrvenv6
brambus qdrvenv7
brambus qdrvenv8
brambus qdrvenv9
brambus qdrvenva
brambus qdrvenvb
brambus qdrvenvc
brambus qdrvenvd
brambus qdrvenve
brambus qdrvenvf
brambus rdrvenv0
brambus rdrvenv1
brambus rdrvenv2
brambus rdrvenv3
brambus rdrvenv4
brambus rdrvenv5
brambus rdrvenv6
brambus rdrvenv7
brambus rdloenv0
brambus rdloenv1
brambus rdloenv2
brambus rdloenv3
brambus rdloenv4
brambus rdloenv5
brambus rdloenv6
brambus rdloenv7
brambus command
brambus accbuf0
brambus accbuf1
brambus accbuf2
brambus accbuf3
brambus accbuf4
brambus accbuf5
brambus accbuf6
brambus accbuf7
brambus acqbuf0
brambus acqbuf1
}


fpgaif "../../src/fpga_master_dir"
set_property master_direction in [ipx::get_bus_abstraction_ports G12 -of_objects [ipx::current_busabs]]
set_property master_direction in [ipx::get_bus_abstraction_ports G11 -of_objects [ipx::current_busabs]]
set_property slave_direction in [ipx::get_bus_abstraction_ports G12 -of_objects [ipx::current_busabs]]
set_property slave_direction in [ipx::get_bus_abstraction_ports G11 -of_objects [ipx::current_busabs]]
ipx::save_bus_definition [ipx::current_busdef]
set_property bus_type_vlnv fpga:user:fpga:1.0 [ipx::current_busabs]
ipx::save_abstraction_definition [ipx::current_busabs]
set_property  ip_repo_paths  ./vivado_project/plip [current_project]
update_ip_catalog
ipx::add_bus_interface fpga [ipx::current_core]
set_property abstraction_type_vlnv fpga:user:fpga_rtl:1.0 [ipx::get_bus_interfaces fpga -of_objects [ipx::current_core]]
set_property bus_type_vlnv fpga:user:fpga:1.0 [ipx::get_bus_interfaces fpga -of_objects [ipx::current_core]]
set_property interface_mode master [ipx::get_bus_interfaces fpga -of_objects [ipx::current_core]]
ipx::add_bus_parameter NUM_READ_OUTSTANDING [ipx::get_bus_interfaces fpga -of_objects [ipx::current_core]]
ipx::add_bus_parameter NUM_WRITE_OUTSTANDING [ipx::get_bus_interfaces fpga -of_objects [ipx::current_core]]
set_property abstraction_type_vlnv fpga:user:fpga_rtl:1.0 [ipx::get_bus_interfaces fpga -of_objects [ipx::current_core]]
set_property bus_type_vlnv fpga:user:fpga:1.0 [ipx::get_bus_interfaces fpga -of_objects [ipx::current_core]]
fpgamap "../../src/fpga_master_dir"

set_property ip_repo_paths {vivado_project/iflocalbus} [current_project]
update_ip_catalog
lbbus lb1
lbbus lb2
lbbus lb3
lbbus lb4
if {0} {
ipx::add_bus_interface lb1 [ipx::current_core]
set_property abstraction_type_vlnv user:user:iflocalbus_rtl:1.0 [ipx::get_bus_interfaces lb1 -of_objects [ipx::current_core]]
set_property bus_type_vlnv user:user:iflocalbus:1.0 [ipx::get_bus_interfaces lb1 -of_objects [ipx::current_core]]

portpin lb1 rdata lb1_rdata
portpin lb1 raddr lb1_raddr
portpin lb1 rden lb1_rden
portpin lb1 wdata lb1_wdata
portpin lb1 waddr lb1_waddr
portpin lb1 wren lb1_wren
portpin lb1 rvalid lb1_rvalid
portpin lb1 clk lb1_clk
portpin lb1 aresetn lb1_aresetn

ipx::add_bus_interface lb2 [ipx::current_core]
set_property abstraction_type_vlnv user:user:iflocalbus_rtl:1.0 [ipx::get_bus_interfaces lb2 -of_objects [ipx::current_core]]
set_property bus_type_vlnv user:user:iflocalbus:1.0 [ipx::get_bus_interfaces lb2 -of_objects [ipx::current_core]]

portpin lb2 rdata lb2_rdata
portpin lb2 raddr lb2_raddr
portpin lb2 rden lb2_rden
portpin lb2 wdata lb2_wdata
portpin lb2 waddr lb2_waddr
portpin lb2 wren lb2_wren
portpin lb2 rvalid lb2_rvalid
portpin lb2 clk lb2_clk
portpin lb2 aresetn lb2_aresetn
}
set_property core_revision 1 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
update_ip_catalog

clkbus pl_sysref
clkbus cfgclk
clkbus dspclk
clkbus clkadc2_300
clkbus clkadc2_600

#ipx::add_bus_interface clkadc2_300 [ipx::current_core]
#set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 [ipx::get_bus_interfaces clkadc2_300 -of_objects [ipx::current_core]]
#set_property bus_type_vlnv xilinx.com:signal:clock:1.0 [ipx::get_bus_interfaces clkadc2_300 -of_objects [ipx::current_core]]
#set_property interface_mode master [ipx::get_bus_interfaces clkadc2_300 -of_objects [ipx::current_core]]
#ipx::remove_bus_parameter FREQ_HZ [ipx::get_bus_interfaces clkadc2_300 -of_objects [ipx::current_core]]
#ipx::add_port_map CLK [ipx::get_bus_interfaces clkadc2_300 -of_objects [ipx::current_core]]
#set_property physical_name clkadc2_300 [ipx::get_port_maps CLK -of_objects [ipx::get_bus_interfaces clkadc2_300 -of_objects [ipx::current_core]]]
#ipx::add_bus_interface clkadc2_600 [ipx::current_core]
#set_property abstraction_type_vlnv xilinx.com:signal:clock_rtl:1.0 [ipx::get_bus_interfaces clkadc2_600 -of_objects [ipx::current_core]]
#set_property bus_type_vlnv xilinx.com:signal:clock:1.0 [ipx::get_bus_interfaces clkadc2_600 -of_objects [ipx::current_core]]
#set_property interface_mode master [ipx::get_bus_interfaces clkadc2_600 -of_objects [ipx::current_core]]
#ipx::remove_bus_parameter FREQ_HZ [ipx::get_bus_interfaces clkadc2_600 -of_objects [ipx::current_core]]
#ipx::add_port_map CLK [ipx::get_bus_interfaces clkadc2_600 -of_objects [ipx::current_core]]
#set_property physical_name clkadc2_600 [ipx::get_port_maps CLK -of_objects [ipx::get_bus_interfaces clkadc2_600 -of_objects [ipx::current_core]]]

ipx::merge_project_changes ports [ipx::current_core]
ipx::update_source_project_archive -component [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]

