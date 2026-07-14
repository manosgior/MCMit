if {0} {
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {axi_clock_converter_0_M_AXI axi_interconnect_1_M01_AXI dspregs_lb}]
apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
                                                          [get_bd_intf_nets axi_clock_converter_0_M_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                          [get_bd_intf_nets axi_interconnect_1_M01_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/plps_0/cfgclk" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                          [get_bd_intf_nets dspregs_lb] {NON_AXI_SIGNALS "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" } \
                                                         ]
}
if {0} {
create_bd_cell -type ip -vlnv xilinx.com:ip:c_counter_binary:12.0 c_counter_binary_0
connect_bd_net [get_bd_pins c_counter_binary_0/CLK] [get_bd_pins plps_0/dspclk]
connect_bd_net -net Q [get_bd_pins c_counter_binary_0/Q]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {axi_clock_converter_0_M_AXI dspregs_lb}]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {Q }]
connect_bd_net -net dspawstate_dbg [get_bd_pins dspregs/awstate_dbg]
connect_bd_net -net dsparstate_dbg [get_bd_pins dspregs/arstate_dbg]
connect_bd_net -net dspwstate_dbg [get_bd_pins dspregs/wstate_dbg]
connect_bd_net -net dsprstate_dbg [get_bd_pins dspregs/rstate_dbg]
connect_bd_net -net dspbstate_dbg [get_bd_pins dspregs/bstate_dbg]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {dspawstate_dbg dsparstate_dbg dspwstate_dbg dsprstate_dbg dspbstate_dbg }]
													 
apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
                                                          [get_bd_nets dsparstate_dbg] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_nets dspawstate_dbg] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_intf_nets axi_clock_converter_0_M_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                          [get_bd_nets dspbstate_dbg] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_intf_nets dspregs_lb] {NON_AXI_SIGNALS "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_nets dsprstate_dbg] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_nets dspwstate_dbg] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_nets Q] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" } \
                                                         ]
connect_bd_net [get_bd_pins rst_psbd_600M/ext_reset_in] [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0]
}
if {0} {


connect_bd_net -net ctrlrstate_dbg [get_bd_pins ctrlregs/rstate_dbg]
connect_bd_net -net ctrlbstate_dbg [get_bd_pins ctrlregs/bstate_dbg]
connect_bd_net -net ctrlawstate_dbg [get_bd_pins ctrlregs/awstate_dbg]
connect_bd_net -net ctrlarstate_dbg [get_bd_pins ctrlregs/arstate_dbg]
connect_bd_net -net ctrlwstate_dbg [get_bd_pins ctrlregs/wstate_dbg]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_nets {ctrlrstate_dbg ctrlbstate_dbg ctrlawstate_dbg ctrlarstate_dbg ctrlwstate_dbg }]
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {ctrlregs_lb axi_interconnect_1_M00_AXI}]
apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
                                                          [get_bd_nets ctrlarstate_dbg] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/cfgclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_nets ctrlawstate_dbg] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/cfgclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_intf_nets axi_interconnect_1_M00_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/plps_0/cfgclk" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                          [get_bd_nets ctrlbstate_dbg] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/cfgclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_intf_nets ctrlregs_lb] {NON_AXI_SIGNALS "Data and Trigger" CLK_SRC "/plps_0/cfgclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_nets ctrlrstate_dbg] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/cfgclk" SYSTEM_ILA "Auto" } \
                                                          [get_bd_nets ctrlwstate_dbg] {PROBE_TYPE "Data and Trigger" CLK_SRC "/plps_0/cfgclk" SYSTEM_ILA "Auto" } \
                                                         ]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_psbd_100M/ext_reset_in]
}
if {0} {
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {axi_interconnect_1_M05_AXI rdrvenv1_ctrl_BRAM_PORTA}]
apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
                                                          [get_bd_intf_nets axi_interconnect_1_M05_AXI] {AXI_R_ADDRESS "Data and Trigger" AXI_R_DATA "Data and Trigger" AXI_W_ADDRESS "Data and Trigger" AXI_W_DATA "Data and Trigger" AXI_W_RESPONSE "Data and Trigger" CLK_SRC "/plps_0/cfgclk" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                          [get_bd_intf_nets rdrvenv1_ctrl_BRAM_PORTA] {NON_AXI_SIGNALS "Data and Trigger" CLK_SRC "/plps_0/cfgclk" SYSTEM_ILA "Auto" } \
                                                         ]
}
if {0} {
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {plps_0_rdrvenv0_ctrl}]
apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
                                                          [get_bd_intf_nets plps_0_rdrvenv0_ctrl] {NON_AXI_SIGNALS "Data and Trigger" CLK_SRC "None (Connect manually)" SYSTEM_ILA "Auto" } \
                                                         ]
connect_bd_net [get_bd_pins plps_0/dspclk] [get_bd_pins system_ila_0/clk]
#apply_bd_automation -rule xilinx.com:bd_rule:board -config { Manual_Source {Auto}}  [get_bd_pins rst_psbd_100M/ext_reset_in]
}
if {0} {
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {rf_data_converter_m30_axis}]
apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
                                                          [get_bd_intf_nets rf_data_converter_m30_axis] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                         ]
apply_bd_automation -rule xilinx.com:bd_rule:board -config { Manual_Source {/zynq_ultra_ps_e_0/pl_resetn0 (ACTIVE_LOW)}}  [get_bd_pins rst_psbd_500M/ext_reset_in]
}
if {0} {
set_property HDL_ATTRIBUTE.DEBUG true [get_bd_intf_nets {plps_0_DAC20_M_AXIS}]
apply_bd_automation -rule xilinx.com:bd_rule:debug -dict [list \
                                                          [get_bd_intf_nets plps_0_DAC20_M_AXIS] {AXIS_SIGNALS "Data and Trigger" CLK_SRC "/plps_0/dspclk" SYSTEM_ILA "Auto" APC_EN "0" } \
                                                         ]
												 }
