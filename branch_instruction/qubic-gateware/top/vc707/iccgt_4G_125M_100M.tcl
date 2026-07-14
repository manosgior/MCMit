create_project tmpproj tmpproj -part xc7vx485tffg1157-1
create_ip -name gtwizard -vendor xilinx.com -library ip -version 3.6 -module_name gtwizard_0
set_property -dict {
CONFIG.gt0_val_align_comma_enable {1111111111}
CONFIG.gt0_val_align_comma_word {Four_Byte_Boundaries}
CONFIG.gt0_val_cc {false}
CONFIG.gt0_val_cc_seq_periodicity {200}
CONFIG.gt0_val_clk_cor_seq_1_1 {00000000}
CONFIG.gt0_val_clk_cor_seq_1_2 {00000000}
CONFIG.gt0_val_clk_cor_seq_1_3 {00000000}
CONFIG.gt0_val_clk_cor_seq_1_4 {00000000}
CONFIG.gt0_val_clk_cor_seq_2_1 {00000000}
CONFIG.gt0_val_clk_cor_seq_2_2 {00000000}
CONFIG.gt0_val_clk_cor_seq_2_3 {00000000}
CONFIG.gt0_val_clk_cor_seq_2_4 {00000000}
CONFIG.gt0_val_clk_cor_seq_2_use {false}
CONFIG.gt0_val_clk_cor_seq_2_use {true}
CONFIG.gt0_val_comma_preset {K28.5}
CONFIG.gt0_val_cpll_fbdiv {4}
CONFIG.gt0_val_cpll_fbdiv_45 {4}
CONFIG.gt0_val_cpll_rxout_div {1}
CONFIG.gt0_val_cpll_txout_div {1}
CONFIG.gt0_val_dec_valid_comma_only {true}
CONFIG.gt0_val_decoding {8B/10B}
CONFIG.gt0_val_dfe_mode {LPM-Auto}
CONFIG.gt0_val_encoding {8B/10B}
CONFIG.gt0_val_port_rxcharisk {true}
CONFIG.gt0_val_port_rxpcommaalignen {true}
CONFIG.gt0_val_port_rxslide {false}
CONFIG.gt0_val_ppm_offset {200}
CONFIG.gt0_val_rx_data_width {32}
CONFIG.gt0_val_rx_int_datawidth {40}
CONFIG.gt0_val_rx_line_rate {4}
CONFIG.gt0_val_rx_reference_clock {125.000}
CONFIG.gt0_val_rx_termination_voltage {Programmable}
CONFIG.gt0_val_rx_cm_trim {100}
CONFIG.gt0_val_rx_cm_trim {100}
CONFIG.gt0_val_rx_termination_voltage {AVTT}
CONFIG.gt0_val_tx_data_width {32}
CONFIG.gt0_val_tx_int_datawidth {40}
CONFIG.gt0_val_tx_line_rate {4}
CONFIG.gt0_val_tx_reference_clock {125.000}
CONFIG.gt_val_rx_pll {CPLL}
CONFIG.gt_val_tx_pll {CPLL}
CONFIG.identical_val_rx_line_rate {4}
CONFIG.identical_val_rx_reference_clock {125.000}
CONFIG.identical_val_tx_line_rate {4}
CONFIG.identical_val_tx_reference_clock {125.000}
CONFIG.gt0_val_rxslide_mode {OFF}
CONFIG.gt0_val_rxusrclk {RXOUTCLK}
} [get_ips gtwizard_0]
generate_target {instantiation_template} [get_files gtwizard_0.xci]
generate_target all [get_files  gtwizard_0.xci]
#CONFIG.gt0_val_rxslide_mode {OFF}
#CONFIG.gt0_val_tx_buffer_bypass_mode {Auto}
#CONFIG.gt0_val_txbuf_en {false}
#CONFIG.gt0_val_txoutclk_source {true}
#CONFIG.gt0_usesharedlogic {1}
#CONFIG.gt0_val_cpll_fbdiv_45 {4}
#CONFIG.gt0_val_dfe_mode {DFE-Auto}
#CONFIG.gt0_val_drp_clock {125}
#CONFIG.gt0_val_port_rxpcommaalignen {true}
#CONFIG.gt0_val_port_rxslide {false}
#CONFIG.gt0_val_rx_buffer_bypass_mode {Auto}
#CONFIG.gt0_val_rxbuf_en {false}
#CONFIG.gt0_val_cpll_fbdiv_45 {5}
#set_property -dict [list CONFIG.gt0_val_rxusrclk {RXOUTCLK} CONFIG.gt0_val_cc {true} CONFIG.gt0_val_ppm_offset {10}] [get_ips gtwizard_0]

