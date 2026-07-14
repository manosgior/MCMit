create_ip -name jesd204_phy -vendor xilinx.com -library ip -version 4.0 -module_name jesd204_phy_8lane
set_property -dict {
    CONFIG.C_LANES {8}
    CONFIG.GT_Line_Rate {10}
    CONFIG.GT_REFCLK_FREQ {500.000}
    CONFIG.DRPCLK_FREQ {125.0}
    CONFIG.C_PLL_SELECTION {3}
    CONFIG.RX_GT_Line_Rate {10}
    CONFIG.RX_GT_REFCLK_FREQ {500.000}
    CONFIG.RX_PLL_SELECTION {3}
    CONFIG.Min_Line_Rate {10} CONFIG.Max_Line_Rate {10}
} [get_ips jesd204_phy_8lane]
#generate_target {instantiation_template} [get_files jesd204_phy_fmc120.xci]
#update_compile_order -fileset sources_1
#generate_target all [get_files  jesd204_phy_fmc120.xci]
#catch { config_ip_cache -export [get_ips -all jesd204_phy_fmc120] }
#export_ip_user_files -of_objects [get_files jesd204_phy_fmc120.xci] -no_script -sync -force -quiet
#create_ip_run [get_files -of_objects [get_fileset sources_1] jesd204_phy_fmc120.xci]
