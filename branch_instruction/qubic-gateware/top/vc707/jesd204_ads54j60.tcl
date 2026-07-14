create_ip -name jesd204 -vendor xilinx.com -library ip -version 7.2 -module_name jesd204_ads54j60
set_property -dict {
    CONFIG.C_NODE_IS_TRANSMIT {0}
    CONFIG.C_LANES {4}
    CONFIG.GT_Line_Rate {10}
    CONFIG.GT_REFCLK_FREQ {500.000}
    CONFIG.DRPCLK_FREQ {125.0}
    CONFIG.C_PLL_SELECTION {3}
    CONFIG.Global_clk_sel {true}
    CONFIG.C_DEFAULT_F {1}
    CONFIG.C_DEFAULT_K {32}
    CONFIG.AXICLK_FREQ {125.0}
} [get_ips jesd204_ads54j60]
#generate_target {instantiation_template} [get_files jesd204_ads54j60.xci]
#generate_target all [get_files  jesd204_ads54j60.xci]
#catch { config_ip_cache -export [get_ips -all jesd204_ads54j60] }
#export_ip_user_files -of_objects [get_files jesd204_ads54j60.xci] -no_script -sync -force -quiet
#create_ip_run [get_files -of_objects [get_fileset sources_1] jesd204_ads54j60.xci]
# launch_runs -jobs 8 jesd204_ads54j60_synth_1
