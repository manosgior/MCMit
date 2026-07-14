create_project tmpproj tmpproj -part xc7vx485tffg1157-1
create_ip -name gtwizard -vendor xilinx.com -library ip -version 3.6 -module_name gtwizard_0
set_property -dict {
CONFIG.gt0_val_align_comma_word {Any_Byte_Boundary}
CONFIG.gt0_val_comma_preset {User_defined}
CONFIG.gt0_val_cpll_fbdiv {4}
CONFIG.gt0_val_cpll_fbdiv_45 {5}
CONFIG.gt0_val_cpll_rxout_div {4}
CONFIG.gt0_val_cpll_txout_div {4}
CONFIG.gt0_val_dec_mcomma_detect {false}
CONFIG.gt0_val_dec_pcomma_detect {false}
CONFIG.gt0_val_port_rxslide {false}
CONFIG.gt0_val_rx_buffer_bypass_mode {Auto}
CONFIG.gt0_val_rx_data_width {20}
CONFIG.gt0_val_rx_int_datawidth {20}
CONFIG.gt0_val_rx_line_rate {1.25}
CONFIG.gt0_val_rx_reference_clock {125.000}
CONFIG.gt0_val_rxbuf_en {false}
CONFIG.gt0_val_rxcomma_deten {false}
CONFIG.gt0_val_rxslide_mode {OFF}
CONFIG.gt0_val_rxusrclk {RXOUTCLK}
CONFIG.gt0_val_tx_buffer_bypass_mode {Auto}
CONFIG.gt0_val_tx_data_width {20}
CONFIG.gt0_val_tx_int_datawidth {20}
CONFIG.gt0_val_tx_line_rate {1.25}
CONFIG.gt0_val_tx_reference_clock {125.000}
CONFIG.gt0_val_txbuf_en {false}
CONFIG.gt0_val_txoutclk_source {true}
CONFIG.gt_val_rx_pll {CPLL}
CONFIG.gt_val_tx_pll {CPLL}
CONFIG.identical_val_rx_line_rate {1.25}
CONFIG.identical_val_rx_reference_clock {125.000}
CONFIG.identical_val_tx_line_rate {1.25}
CONFIG.identical_val_tx_reference_clock {125.000}
} [get_ips gtwizard_0]
generate_target {instantiation_template} [get_files gtwizard_0.xci]
generate_target all [get_files  gtwizard_0.xci]

