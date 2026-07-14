#proc bramwithctrl {bramname Awidth Adepth Bwidth latency} {
#	incr $Awidth 0
#	incr $Adepth 0
#	incr $Bwidth 0
#	incr $latency 0
#	create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 ${bramname}_ctrl
#	create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 ${bramname}_mem
#	set_property -dict [list CONFIG.READ_LATENCY ${latency} CONFIG.SINGLE_PORT_BRAM {1} CONFIG.DATA_WIDTH ${Awidth}] [get_bd_cells ${bramname}_ctrl]
#	set_property -dict [list CONFIG.use_bram_block {Stand_Alone} CONFIG.Memory_Type {True_Dual_Port_RAM} CONFIG.Write_Depth_A ${Adepth} CONFIG.Write_Width_A ${Awidth} CONFIG.Read_Width_A ${Awidth} CONFIG.Write_Width_B ${Bwidth} CONFIG.Read_Width_B ${Bwidth} CONFIG.Enable_32bit_Address {true} CONFIG.Use_Byte_Write_Enable {true} CONFIG.Byte_Size {8} CONFIG.Use_RSTA_Pin {false} CONFIG.Use_RSTB_Pin {false} CONFIG.Enable_A {Always_Enabled} CONFIG.Enable_B {Always_Enabled} CONFIG.EN_SAFETY_CKT {false}] [get_bd_cells ${bramname}_mem]
#}


create_bd_design "psbd"
open_bd_design {./vivado_project/psbd.srcs/sources_1/bd/psbd/psbd.bd}
source ../../src/ps.tcl
source ../../src/rfdc.tcl
set_property  ip_repo_paths  {./vivado_project/plip ./vivado_project/iflocalbus} [current_project]
update_ip_catalog -rebuild
create_bd_cell -type ip -vlnv user.org:user:plps:1.0 plps_0
create_bd_intf_port -mode Master -vlnv fpga:user:fpga_rtl:1.0 fpga
create_bd_cell -type ip -vlnv user.org:user:axi4_lb:1.0 cfgregs
create_bd_cell -type ip -vlnv user.org:user:axi4_lb:1.0 bramctrl
create_bd_cell -type ip -vlnv user.org:user:axi4_lb:1.0 dspregs
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_1
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 axi_clock_converter_0
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout00
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout01
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout02
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout03
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout10
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout11
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout12
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout13
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout20
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout21
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout22
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout23
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout30
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout31
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout32
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vout33
create_bd_intf_port -mode Slave -vlnv xilinx.com:display_usp_rf_data_converter:diff_pins_rtl:1.0 sysref_in
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 dac2_clk
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 adc2_clk
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin00
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin01
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin02
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin03
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin10
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin11
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin12
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin13
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin20
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin21
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin22
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin23
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin30
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin31
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin32
create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_analog_io_rtl:1.0 vin33
set_property -dict [list CONFIG.NUM_MI {4}] [get_bd_cells axi_interconnect_1]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_lpd_aclk] [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
connect_bd_net [get_bd_pins axi_interconnect_1/S00_ACLK] [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
connect_bd_net [get_bd_pins axi_interconnect_1/ACLK] [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_LPD] [get_bd_intf_pins axi_interconnect_1/S00_AXI]
set_property -dict [list CONFIG.FREQ_HZ {100000000}] [get_bd_pins plps_0/cfgclk]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/dspclk]
set_property -dict [list CONFIG.CLK_FREQ_HZ {100000000}] [get_bd_cells cfgregs]
set_property -dict [list CONFIG.CLK_FREQ_HZ {100000000}] [get_bd_cells bramctrl]
set_property -dict [list CONFIG.CLK_FREQ_HZ {500000000}] [get_bd_cells dspregs]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC00_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC01_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC02_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC03_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC10_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC11_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC12_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC13_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC20_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC21_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC22_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC23_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC30_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC31_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC32_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/DAC33_M_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC00_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC01_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC02_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC03_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC10_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC11_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC12_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC13_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC20_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC21_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC22_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC23_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC30_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC31_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC32_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/ADC33_S_AXIS_ACLK]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins dspregs/axi_aclk]
set_property -dict [list CONFIG.FREQ_HZ {100000000}] [get_bd_pins cfgregs/axi_aclk]
set_property -dict [list CONFIG.FREQ_HZ {100000000}] [get_bd_pins bramctrl/axi_aclk]
set_property -dict [list CONFIG.FREQ_HZ {250000000}] [get_bd_pins plps_0/clkadc3_300]
set_property -dict [list CONFIG.FREQ_HZ {500000000}] [get_bd_pins plps_0/clkadc3_600]
set_property -dict [list CONFIG.FREQ_HZ {625000}] [get_bd_pins plps_0/pl_sysref]



set_property -dict [list CONFIG.ADDR_WIDTH {16} CONFIG.ID_WIDTH {16}] [get_bd_cells dspregs]
set_property -dict [list CONFIG.ADDR_WIDTH {16} CONFIG.ID_WIDTH {16}] [get_bd_cells cfgregs]
set_property -dict [list CONFIG.ADDR_WIDTH {26} CONFIG.ID_WIDTH {16}] [get_bd_cells bramctrl]
#set_property -dict [list CONFIG.LB_DATAWIDTH {32} CONFIG.LB_ADDRWIDTH {14} CONFIG.DAC_AXIS_DATAWIDTH {256} CONFIG.ADC_AXIS_DATAWIDTH {64} CONFIG.BRAMTOHOST_ADDRWIDTH {17} CONFIG.BRAMTOHOST_DATAWIDTH {64} CONFIG.BRAMFROMHOST_ADDRWIDTH {15} CONFIG.BRAMFROMHOST_DATAWIDTH {256}] [get_bd_cells plps_0]
set_property -dict [list CONFIG.LB1_DATAWIDTH {32} CONFIG.LB1_ADDRWIDTH {14} CONFIG.LB2_DATAWIDTH {32} CONFIG.LB2_ADDRWIDTH {14} CONFIG.LB3_DATAWIDTH {32} CONFIG.LB3_ADDRWIDTH {26} CONFIG.LB4_DATAWIDTH {32} CONFIG.LB4_ADDRWIDTH {20} CONFIG.DAC_AXIS_DATAWIDTH {256} CONFIG.ADC_AXIS_DATAWIDTH {64}] [get_bd_cells plps_0]
#CONFIG.BRAMTOHOST_ADDRWIDTH {32} CONFIG.BRAMTOHOST_DATAWIDTH {64} CONFIG.BRAMFROMHOST_ADDRWIDTH {32} CONFIG.BRAMFROMHOST_DATAWIDTH {512}


connect_bd_intf_net [get_bd_intf_pins axi_interconnect_1/M00_AXI] [get_bd_intf_pins cfgregs/axi]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_1/M01_AXI] [get_bd_intf_pins axi_clock_converter_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_1/M02_AXI] [get_bd_intf_pins rf_data_converter/s_axi]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_1/M03_AXI] [get_bd_intf_pins bramctrl/axi]


connect_bd_intf_net [get_bd_intf_pins axi_clock_converter_0/M_AXI] [get_bd_intf_pins dspregs/axi]
#connect_bd_intf_net [get_bd_intf_pins rf_data_converter/adc3_clk] [get_bd_intf_ports adc3_clk]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/dac2_clk] [get_bd_intf_ports dac2_clk]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/adc2_clk] [get_bd_intf_ports adc2_clk]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/sysref_in] [get_bd_intf_ports sysref_in]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin00] [get_bd_intf_ports vin00]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin01] [get_bd_intf_ports vin01]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin02] [get_bd_intf_ports vin02]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin03] [get_bd_intf_ports vin03]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin10] [get_bd_intf_ports vin10]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin11] [get_bd_intf_ports vin11]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin12] [get_bd_intf_ports vin12]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin13] [get_bd_intf_ports vin13]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin20] [get_bd_intf_ports vin20]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin21] [get_bd_intf_ports vin21]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin22] [get_bd_intf_ports vin22]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin23] [get_bd_intf_ports vin23]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin30] [get_bd_intf_ports vin30]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin31] [get_bd_intf_ports vin31]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin32] [get_bd_intf_ports vin32]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vin33] [get_bd_intf_ports vin33]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout00] [get_bd_intf_ports vout00]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout01] [get_bd_intf_ports vout01]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout02] [get_bd_intf_ports vout02]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout03] [get_bd_intf_ports vout03]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout10] [get_bd_intf_ports vout10]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout11] [get_bd_intf_ports vout11]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout12] [get_bd_intf_ports vout12]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout13] [get_bd_intf_ports vout13]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout20] [get_bd_intf_ports vout20]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout21] [get_bd_intf_ports vout21]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout22] [get_bd_intf_ports vout22]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout23] [get_bd_intf_ports vout23]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout30] [get_bd_intf_ports vout30]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout31] [get_bd_intf_ports vout31]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout32] [get_bd_intf_ports vout32]
connect_bd_intf_net [get_bd_intf_pins rf_data_converter/vout33] [get_bd_intf_ports vout33]


connect_bd_intf_net [get_bd_intf_pins plps_0/ADC00_S_AXIS] [get_bd_intf_pins rf_data_converter/m00_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC01_S_AXIS] [get_bd_intf_pins rf_data_converter/m01_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC02_S_AXIS] [get_bd_intf_pins rf_data_converter/m02_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC03_S_AXIS] [get_bd_intf_pins rf_data_converter/m03_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC10_S_AXIS] [get_bd_intf_pins rf_data_converter/m10_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC11_S_AXIS] [get_bd_intf_pins rf_data_converter/m11_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC12_S_AXIS] [get_bd_intf_pins rf_data_converter/m12_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC13_S_AXIS] [get_bd_intf_pins rf_data_converter/m13_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC20_S_AXIS] [get_bd_intf_pins rf_data_converter/m20_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC21_S_AXIS] [get_bd_intf_pins rf_data_converter/m21_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC22_S_AXIS] [get_bd_intf_pins rf_data_converter/m22_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC23_S_AXIS] [get_bd_intf_pins rf_data_converter/m23_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC30_S_AXIS] [get_bd_intf_pins rf_data_converter/m30_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC31_S_AXIS] [get_bd_intf_pins rf_data_converter/m31_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC32_S_AXIS] [get_bd_intf_pins rf_data_converter/m32_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/ADC33_S_AXIS] [get_bd_intf_pins rf_data_converter/m33_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC00_M_AXIS] [get_bd_intf_pins rf_data_converter/s00_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC01_M_AXIS] [get_bd_intf_pins rf_data_converter/s01_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC02_M_AXIS] [get_bd_intf_pins rf_data_converter/s02_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC03_M_AXIS] [get_bd_intf_pins rf_data_converter/s03_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC10_M_AXIS] [get_bd_intf_pins rf_data_converter/s10_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC11_M_AXIS] [get_bd_intf_pins rf_data_converter/s11_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC12_M_AXIS] [get_bd_intf_pins rf_data_converter/s12_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC13_M_AXIS] [get_bd_intf_pins rf_data_converter/s13_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC20_M_AXIS] [get_bd_intf_pins rf_data_converter/s20_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC21_M_AXIS] [get_bd_intf_pins rf_data_converter/s21_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC22_M_AXIS] [get_bd_intf_pins rf_data_converter/s22_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC23_M_AXIS] [get_bd_intf_pins rf_data_converter/s23_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC30_M_AXIS] [get_bd_intf_pins rf_data_converter/s30_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC31_M_AXIS] [get_bd_intf_pins rf_data_converter/s31_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC32_M_AXIS] [get_bd_intf_pins rf_data_converter/s32_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/DAC33_M_AXIS] [get_bd_intf_pins rf_data_converter/s33_axis]
connect_bd_intf_net [get_bd_intf_pins plps_0/fpga] [get_bd_intf_ports fpga]
connect_bd_intf_net [get_bd_intf_pins plps_0/lb1] [get_bd_intf_pins cfgregs/lb]
connect_bd_intf_net [get_bd_intf_pins plps_0/lb2] [get_bd_intf_pins dspregs/lb]
connect_bd_intf_net [get_bd_intf_pins plps_0/lb3] [get_bd_intf_pins bramctrl/lb]
connect_bd_net [get_bd_pins plps_0/aresetn] [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0]
connect_bd_net [get_bd_pins plps_0/cfgclk] [get_bd_pins axi_clock_converter_0/s_axi_aclk]
connect_bd_net [get_bd_pins plps_0/cfgclk] [get_bd_pins axi_interconnect_1/M00_ACLK]
connect_bd_net [get_bd_pins plps_0/cfgclk] [get_bd_pins axi_interconnect_1/M01_ACLK]
connect_bd_net [get_bd_pins plps_0/cfgclk] [get_bd_pins axi_interconnect_1/M02_ACLK]
connect_bd_net [get_bd_pins plps_0/cfgclk] [get_bd_pins axi_interconnect_1/M03_ACLK] 
connect_bd_net [get_bd_pins plps_0/cfgclk] [get_bd_pins cfgregs/axi_aclk]
connect_bd_net [get_bd_pins plps_0/cfgclk] [get_bd_pins bramctrl/axi_aclk]
connect_bd_net [get_bd_pins plps_0/cfgclk] [get_bd_pins rf_data_converter/s_axi_aclk]
connect_bd_net [get_bd_pins plps_0/cfgresetn0] [get_bd_pins axi_interconnect_1/M00_ARESETN] 
connect_bd_net [get_bd_pins plps_0/cfgresetn1] [get_bd_pins axi_interconnect_1/M01_ARESETN]
connect_bd_net [get_bd_pins plps_0/cfgresetn2] [get_bd_pins axi_interconnect_1/M02_ARESETN]
connect_bd_net [get_bd_pins plps_0/cfgresetn3] [get_bd_pins axi_interconnect_1/M03_ARESETN]
connect_bd_net [get_bd_pins plps_0/cfgresetn4] [get_bd_pins cfgregs/axi_aresetn]
connect_bd_net [get_bd_pins plps_0/cfgresetn5] [get_bd_pins bramctrl/axi_aresetn]
connect_bd_net [get_bd_pins plps_0/cfgresetn6] [get_bd_pins axi_clock_converter_0/s_axi_aresetn]
connect_bd_net [get_bd_pins plps_0/cfgresetn7] [get_bd_pins rf_data_converter/s_axi_aresetn]
connect_bd_net [get_bd_pins plps_0/clk_dac0] [get_bd_pins rf_data_converter/clk_dac0]
connect_bd_net [get_bd_pins plps_0/clk_dac1] [get_bd_pins rf_data_converter/clk_dac1]
connect_bd_net [get_bd_pins plps_0/clk_dac2] [get_bd_pins rf_data_converter/clk_dac2]
connect_bd_net [get_bd_pins plps_0/clk_dac3] [get_bd_pins rf_data_converter/clk_dac3]
connect_bd_net [get_bd_pins plps_0/clk_adc0] [get_bd_pins rf_data_converter/clk_adc0]
connect_bd_net [get_bd_pins plps_0/clk_adc1] [get_bd_pins rf_data_converter/clk_adc1]
connect_bd_net [get_bd_pins plps_0/clk_adc2] [get_bd_pins rf_data_converter/clk_adc2]
connect_bd_net [get_bd_pins plps_0/clk_adc3] [get_bd_pins rf_data_converter/clk_adc3]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC00_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC01_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC02_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC03_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC10_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC11_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC12_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC13_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC20_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC21_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC22_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC23_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC30_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC31_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC32_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/ADC33_S_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins rf_data_converter/m3_axis_aclk]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins rf_data_converter/m2_axis_aclk]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins rf_data_converter/m1_axis_aclk]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins rf_data_converter/m0_axis_aclk]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins axi_clock_converter_0/m_axi_aclk]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins dspregs/axi_aclk]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC00_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC01_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC02_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC03_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC10_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC11_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC12_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC13_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC20_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC21_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC22_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC23_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC30_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC31_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC32_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins plps_0/DAC33_M_AXIS_ACLK]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins rf_data_converter/s0_axis_aclk]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins rf_data_converter/s1_axis_aclk]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins rf_data_converter/s2_axis_aclk]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins rf_data_converter/s3_axis_aclk]
connect_bd_net [get_bd_pins plps_0/dspresetn0] [get_bd_pins dspregs/axi_aresetn] 
connect_bd_net [get_bd_pins plps_0/dspresetn1] [get_bd_pins axi_clock_converter_0/m_axi_aresetn]
connect_bd_net [get_bd_pins plps_0/dspresetn2] [get_bd_pins plps_0/DAC00_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn3] [get_bd_pins plps_0/DAC01_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn4] [get_bd_pins plps_0/DAC02_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn5] [get_bd_pins plps_0/DAC03_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn6] [get_bd_pins plps_0/DAC10_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn7] [get_bd_pins plps_0/DAC11_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn8] [get_bd_pins plps_0/DAC12_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn9] [get_bd_pins plps_0/DAC13_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn10] [get_bd_pins plps_0/DAC20_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn11] [get_bd_pins plps_0/DAC21_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn12] [get_bd_pins plps_0/DAC22_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn13] [get_bd_pins plps_0/DAC23_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn14] [get_bd_pins plps_0/DAC30_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn15] [get_bd_pins plps_0/DAC31_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn16] [get_bd_pins plps_0/DAC32_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn17] [get_bd_pins plps_0/DAC33_M_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn18] [get_bd_pins plps_0/ADC00_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn19] [get_bd_pins plps_0/ADC01_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn20] [get_bd_pins plps_0/ADC02_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn21] [get_bd_pins plps_0/ADC03_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn22] [get_bd_pins plps_0/ADC10_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn23] [get_bd_pins plps_0/ADC11_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn24] [get_bd_pins plps_0/ADC12_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn25] [get_bd_pins plps_0/ADC13_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn26] [get_bd_pins plps_0/ADC20_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn27] [get_bd_pins plps_0/ADC21_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn28] [get_bd_pins plps_0/ADC22_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn29] [get_bd_pins plps_0/ADC23_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn30] [get_bd_pins plps_0/ADC30_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn31] [get_bd_pins plps_0/ADC31_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn32] [get_bd_pins plps_0/ADC32_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn33] [get_bd_pins plps_0/ADC33_S_AXIS_ARESETN]
connect_bd_net [get_bd_pins plps_0/dspresetn34] [get_bd_pins rf_data_converter/s0_axis_aresetn]
connect_bd_net [get_bd_pins plps_0/dspresetn35] [get_bd_pins rf_data_converter/s1_axis_aresetn]
connect_bd_net [get_bd_pins plps_0/dspresetn36] [get_bd_pins rf_data_converter/s2_axis_aresetn]
connect_bd_net [get_bd_pins plps_0/dspresetn37] [get_bd_pins rf_data_converter/s3_axis_aresetn]
connect_bd_net [get_bd_pins plps_0/dspresetn38] [get_bd_pins rf_data_converter/m3_axis_aresetn]
connect_bd_net [get_bd_pins plps_0/dspresetn39] [get_bd_pins rf_data_converter/m2_axis_aresetn]
connect_bd_net [get_bd_pins plps_0/dspresetn40] [get_bd_pins rf_data_converter/m1_axis_aresetn]
connect_bd_net [get_bd_pins plps_0/dspresetn41] [get_bd_pins rf_data_converter/m0_axis_aresetn]
connect_bd_net [get_bd_pins plps_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
connect_bd_net [get_bd_pins plps_0/psresetn0] [get_bd_pins axi_interconnect_1/ARESETN]
connect_bd_net [get_bd_pins plps_0/psresetn1] [get_bd_pins axi_interconnect_1/S00_ARESETN]
connect_bd_net [get_bd_pins plps_0/pl_sysref] [get_bd_pins rf_data_converter/user_sysref_dac]
connect_bd_net [get_bd_pins plps_0/pl_sysref] [get_bd_pins rf_data_converter/user_sysref_adc]



source bdila.tcl
assign_bd_address

validate_bd_design
save_bd_design
make_wrapper -files [get_files ./vivado_project/psbd/psbd.srcs/sources_1/bd/psbd/psbd.bd] -top
add_files -norecurse ./vivado_project/psbd/psbd.gen/sources_1/bd/psbd/hdl/psbd_wrapper.v

 
