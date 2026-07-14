#create_clock -period 8.000 -name sgmiiclk -waveform {0.000 4.000} [get_pins qubichw_config/hw\\.vc707\\.sgmiiclk_q0_p]
#create_clock -period 5.000 -name sysclk -waveform {0.000 2.500} [get_pins qubichw_config/hw\\.vc707\\.sysclk]
#create_clock -period 4.000 -name si5324clk -waveform {0.000 2.000} [get_pins qubichw_config/hw\\.vc707\\.si5324_out_c_p]
#create_clock -period 8.000 -name userclk -waveform {0.000 4.000} [get_pins qubichw_config/hw\\.vc707\\.user_clock_p]
#create_clock -period 16.000 -name ethclk -waveform {0.000 8.000} [get_pins qubichw_config/sgmii_ethernet_pcs_pma/bufgtxoutclk/I]

create_clock -period 8.000 -name sgmiiclk -waveform {0.000 4.00} [get_ports {fpga\\.AH7}]
create_clock -period 5.000 -name sysclk -waveform {0.000 2.500} [get_ports  {fpga\\.E19}]
create_clock -period 8.000 -name si5324clk -waveform {0.000 4.000} [get_ports  {fpga\\.AD8}]
create_clock -period 8.000 -name userclk -waveform {0.000 4.000} [get_ports  {fpga\\.AK34}]
create_clock -period 8.000 -name smamgtclk -waveform {0.000 4.000} [get_ports  {fpga\\.AK8}]
#create_clock -period 16.000 -name ethclk -waveform {0.000 8.000} [get_pins qubichw_config/sgmii_ethernet_pcs_pma/bufgtxoutclk/I]
#create_clock -period 16.000 -name ethclk -waveform {0.000 8.000} [get_pins qubichw_config/sgmii_ethernet_pcs_pma/gig_ethernet_pcs_pma_0/txoutclk]
create_clock -period 4.000 -name fmc1dclk2 -waveform {0.000 2.000} [get_ports {fpga\\.L39}]
create_clock -period 2.000 -name fmc1refclk8 -waveform {0.000 1.000} [get_ports {fpga\\.A10}]
create_clock -period 8.000 -name fmc1refclk10 -waveform {0.000 4.000} [get_ports {fpga\\.E10}]
create_clock -period 4.000 -name fmc2dclk2 -waveform {0.000 2.000} [get_ports {fpga\\.AF39}]
create_clock -period 2.000 -name fmc2refclk8 -waveform {0.000 1.000} [get_ports {fpga\\.K8}]
create_clock -period 8.000 -name fmc2refclk10 -waveform {0.000 4.000} [get_ports {fpga\\.T8}]
create_clock -period 500.000 -name iicsclk -waveform {0.000 250.000} [get_ports {fpga\\.AT35}]

create_generated_clock -name clk100 -source [get_ports {fpga\\.E19}] -divide_by 10 -multiply_by 5 [get_pins qubichw_config/sysclkmmcm/bufgclk100/I]
create_generated_clock -name clk125 -source [get_ports {fpga\\.E19}] -divide_by 8 -multiply_by 5 [get_pins qubichw_config/sysclkmmcm/bufgclk125/I]
create_generated_clock -name clk200 -source [get_ports {fpga\\.E19}] -divide_by 5 -multiply_by 5 [get_pins qubichw_config/sysclkmmcm/bufgclk200/I]
create_generated_clock -name clk250 -source [get_ports {fpga\\.E19}] -divide_by 4 -multiply_by 5 [get_pins qubichw_config/sysclkmmcm/bufgclk250/I]
create_generated_clock -name eth125 -source [get_pins qubichw_config/sgmii_ethernet_pcs_pma/bufgtxoutclk/I] -divide_by 8 -multiply_by 16 [get_pins qubichw_config/sgmii_ethernet_pcs_pma/bufgclk125/I]
create_generated_clock -name eth62_5 -source [get_pins qubichw_config/sgmii_ethernet_pcs_pma/bufgtxoutclk/I] -divide_by 16 -multiply_by 16 [get_pins qubichw_config/sgmii_ethernet_pcs_pma/bufgclk62_5/I]
#create_generated_clock -name si5324clkdiv2 -source [get_pins {qubichw_config/si5324divbufgce/I}] -divide_by 2 [get_pins qubichw_config/si5324divbufgce/O]
#create_generated_clock -name smamgtclkdiv2 -source [get_pins {qubichw_config/smamgtdivbufgce/I}] -divide_by 2 [get_pins qubichw_config/smamgtdivbufgce/O]
#-edges {1,2,3} 
#-edges {1,2,3}

#set_clock_groups -name qubic -asynchronous -group [get_clocks -include_generated_clocks sgmiiclk] -group [get_clocks -include_generated_clocks sysclk] -group [get_clocks -include_generated_clocks si5324clk] -group [get_clocks -include_generated_clocks smamgtclk] -group [get_clocks -include_generated_clocks userclk] -group [get_clocks -include_generated_clocks fmc1dclk2] -group [get_clocks -include_generated_clocks fmc1refclk8] -group [get_clocks -include_generated_clocks fmc1refclk10] -group [get_clocks -include_generated_clocks fmc2dclk2] -group [get_clocks -include_generated_clocks fmc2refclk8] -group [get_clocks -include_generated_clocks fmc2refclk10] -group [get_clocks -include_generated_clocks iicsclk] -group [get_clocks -include_generated_clocks eth125]

#set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets qubichw_config/gticc_common_114/QPLLOUTCLK]
#set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets qubichw_config/gticc_gt_smasfp/TXOUTCLK]

set_max_delay -datapath_only -from [get_pins qubichw_config/*/*done*/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[*]/D}] 16.0
#set_max_delay -from [get_pins qubichw_config/fmcdac2init/done_reg/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[18]/D}] 16.0
#set_max_delay -from [get_pins qubichw_config/fmcdacinit/done_reg/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[17]/D}] 16.0
#set_max_delay -from [get_pins qubichw_config/fmclmkadcinit/done_reg/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[12]/D}] 16.0
#set_max_delay -from [get_pins qubichw_config/gticc_gt_sfp/done_r_reg/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[19]/D}] 16.0
#set_max_delay -from [get_pins qubichw_config/gticc_gt_smasfp/done_r_reg/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[21]/D}] 16.0
#set_max_delay -from [get_pins qubichw_config/i2cinit/done_reg_replica/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[9]/D}] 16.0
#set_max_delay -from [get_pins qubichw_config/i2cmaster/resetdone_r_reg/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[7]/D}] 16.0
#set_max_delay -from [get_pins qubichw_config/mdiomasterinit/done_reg/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[8]/D}] 16.0

set_max_delay -datapath_only -from [get_pins {qubichw_config/chainreset/resetout_r_reg[*]/C}] -to [get_pins {qubichw_config/*/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0
set_max_delay -datapath_only -from [get_pins {qubichw_config/chainreset/resetout_r_reg[*]/C}] -to [get_pins {qubichw_config/phyresetxdomain/datasr1_reg[*]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[10]/C}] -to [get_pins {qubichw_config/loadmacipreset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[12]/C}] -to [get_pins {qubichw_config/fmclmkadcinitareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[17]/C}] -to [get_pins {qubichw_config/fmcdacinitareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[18]/C}] -to [get_pins {qubichw_config/fmcdac2initareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[19]/C}] -to [get_pins {qubichw_config/sfpiccresetareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[21]/C}] -to [get_pins {qubichw_config/smasfpiccresetareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[4]/C}] -to [get_pins {qubichw_config/sgmiiareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[6]/C}] -to [get_pins {qubichw_config/uartlbareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[7]/C}] -to [get_pins {qubichw_config/i2cmasterareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[7]/C}] -to [get_pins {qubichw_config/i2cresetareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/chainreset/resetout_r_reg[8]/C}] -to [get_pins {qubichw_config/mdioresetareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0

set_max_delay -datapath_only -from [get_pins {qubichw_config/uartlb64/lbwcmd_r_reg[*]/C}] -to [get_pins {qubichw_config/uarthwresetareset/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0
set_max_delay -datapath_only -from [get_pins {qubichw_config/udplb64/lbrxdata_r_reg[*]/C}] -to [get_pins {qubichw_config/*/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/udplb64/lbrxdata_r_reg[54]/C}] -to [get_pins {qubichw_config/bufreadtestarset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/udplb64/lbrxdata_r_reg[57]/C}] -to [get_pins {qubichw_config/adc0bufareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
#set_max_delay -from [get_pins {qubichw_config/udplb64/lbrxdata_r_reg[57]/C}] -to [get_pins {qubichw_config/hwresetareset/genblk1[0].areset_i/reset_p_reg/PRE}] 16.0
set_max_delay -datapath_only -from [get_pins {qubichw_config/eepromrd_reg[*]/C}] -to [get_pins {qubichw_config/chainresetareset/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0

set_max_delay -datapath_only -from [get_pins {qubichw_config/donexdomainethclk/data_latch_reg[*]/C}] -to [get_pins {qubichw_config/donexdomainethclk/data_pipe_reg[*]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/donexdomainethclk/data_latch_reg[12]/C}] -to [get_pins {qubichw_config/donexdomainethclk/data_pipe_reg[12]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/donexdomainethclk/data_latch_reg[16]/C}] -to [get_pins {qubichw_config/donexdomainethclk/data_pipe_reg[16]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/donexdomainethclk/data_latch_reg[17]/C}] -to [get_pins {qubichw_config/donexdomainethclk/data_pipe_reg[17]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/donexdomainethclk/data_latch_reg[18]/C}] -to [get_pins {qubichw_config/donexdomainethclk/data_pipe_reg[18]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/donexdomainethclk/data_latch_reg[19]/C}] -to [get_pins {qubichw_config/donexdomainethclk/data_pipe_reg[19]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/donexdomainethclk/data_latch_reg[21]/C}] -to [get_pins {qubichw_config/donexdomainethclk/data_pipe_reg[21]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/donexdomainethclk/data_latch_reg[2]/C}] -to [get_pins {qubichw_config/donexdomainethclk/data_pipe_reg[2]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/donexdomainethclk/data_latch_reg[4]/C}] -to [get_pins {qubichw_config/donexdomainethclk/data_pipe_reg[4]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/donexdomainethclk/data_latch_reg[6]/C}] -to [get_pins {qubichw_config/donexdomainethclk/data_pipe_reg[6]/D}] 16.0

set_max_delay -datapath_only -from [get_pins {qubichw_config/gen_fcnt[*].freq_count/gray1_reg[*]/C}] -to [get_pins {qubichw_config/gen_fcnt[*].freq_count/gray2_reg[*]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[10].freq_count/gray1_reg[1]/C}] -to [get_pins {qubichw_config/gen_fcnt[10].freq_count/gray2_reg[1]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[10].freq_count/gray1_reg[2]/C}] -to [get_pins {qubichw_config/gen_fcnt[10].freq_count/gray2_reg[2]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[10].freq_count/gray1_reg[3]/C}] -to [get_pins {qubichw_config/gen_fcnt[10].freq_count/gray2_reg[3]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[17].freq_count/gray1_reg[0]/C}] -to [get_pins {qubichw_config/gen_fcnt[17].freq_count/gray2_reg[0]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[17].freq_count/gray1_reg[1]/C}] -to [get_pins {qubichw_config/gen_fcnt[17].freq_count/gray2_reg[1]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[17].freq_count/gray1_reg[2]/C}] -to [get_pins {qubichw_config/gen_fcnt[17].freq_count/gray2_reg[2]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[17].freq_count/gray1_reg[3]/C}] -to [get_pins {qubichw_config/gen_fcnt[17].freq_count/gray2_reg[3]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[6].freq_count/gray1_reg[0]/C}] -to [get_pins {qubichw_config/gen_fcnt[6].freq_count/gray2_reg[0]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[6].freq_count/gray1_reg[1]/C}] -to [get_pins {qubichw_config/gen_fcnt[6].freq_count/gray2_reg[1]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[6].freq_count/gray1_reg[2]/C}] -to [get_pins {qubichw_config/gen_fcnt[6].freq_count/gray2_reg[2]/D}] 16.0
#set_max_delay -from [get_pins {qubichw_config/gen_fcnt[6].freq_count/gray1_reg[3]/C}] -to [get_pins {qubichw_config/gen_fcnt[6].freq_count/gray2_reg[3]/D}] 16.0

set_max_delay -datapath_only -from [get_pins qubichw_config/bufreadtest/full_r_reg/C] -to [get_pins {qubichw_config/bufreadtestfulllatch/datasr1_reg[*]/D}] 16.0
set_max_delay -datapath_only -from [get_pins {qubichw_config/ethareset/genblk1[*].areset_i/reset_sr_reg[*]/C}] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[*]/D}] 16.0
set_max_delay -datapath_only -from [get_pins qubichw_config/sgmii_ethernet_pcs_pma/gig_ethernet_pcs_pma*/inst/sync_block_tx_reset_done/data_sync_reg*/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[*]/D}] 16.0
set_max_delay -datapath_only -from [get_pins qubichw_config/sgmii_ethernet_pcs_pma/gig_ethernet_pcs_pma*/inst/sync_block_rx_reset_done/data_sync_reg*/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[*]/D}] 16.0
set_max_delay -datapath_only -from [get_pins qubichw_config/maciploaded*/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[*]/D}] 16.0

set_max_delay -datapath_only -from [get_pins qubichw_config/donexdomainethclk/flag_xdomain/flagtoggle_clk1_reg/C] -to [get_pins {qubichw_config/donexdomainethclk/flag_xdomain/sync1_clk2_reg[*]/D}] 16.0
set_max_delay -datapath_only -from [get_pins {qubichw_config/udplb64/lbrxdata_r_reg[*]*/C}] -to [get_pins {qubichw_config/bufreadtestarset/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0
set_max_delay -datapath_only -from [get_pins {qubichw_config/udplb64/lbrxdata_r_reg[*]*/C}] -to [get_pins {qubichw_config/adc0bufareset/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0
set_max_delay -datapath_only -from [get_pins {qubichw_config/udplb64/lbrxdata_r_reg[*]*/C}] -to [get_pins {qubichw_config/hwresetareset/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0
set_max_delay -datapath_only -from [get_pins qubichw_config/udplb64/lbrxdv_r_reg*/C] -to [get_pins {qubichw_config/bufreadtestarset/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0
set_max_delay -datapath_only -from [get_pins qubichw_config/udplb64/lbrxdv_r_reg*/C] -to [get_pins {qubichw_config/adc0bufareset/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0
set_max_delay -datapath_only -from [get_pins qubichw_config/udplb64/lbrxdv_r_reg*/C] -to [get_pins {qubichw_config/hwresetareset/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0
set_max_delay -datapath_only -from [get_pins qubichw_config/uartlb64/lbwvalid_r_reg/C] -to [get_pins {qubichw_config/uarthwresetareset/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0
set_max_delay -datapath_only -from [get_pins {reg_loopreset_reg[*]/C}] -to [get_pins {qubichw_config/chainresetareset/genblk1[*].areset_i/reset_p_reg/PRE}] 16.0

#set_max_delay -from [get_pins qubichw_config/jesdfmc120*/jesd204_phy_fmc120/inst/jesd204_phy_block_i/tx_reset_done_r_reg/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[*]/D}] 16.0
set_max_delay -datapath_only -from [get_pins qubichw_config/jesdfmc120_2/jesd204_phy_fmc120/inst/jesd204_phy_block_i/rx_reset_done_r_reg/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[*]/D}] 32.0
set_max_delay -datapath_only -from [get_pins qubichw_config/jesdfmc120_2/jesd204_phy_fmc120/inst/jesd204_phy_block_i/tx_reset_done_r_reg/C] -to [get_pins {qubichw_config/donecriteriaxdomain/datasr1_reg[*]/D}] 32.0

set_max_delay -from [get_pins qubichw_config/jesdreset1xdomain/flagtoggle_clk1*/C] -to [get_pins {qubichw_config/jesdreset1xdomain/sync1_clk2_reg*/D}] 16.0
set_max_delay -from [get_pins qubichw_config/*/C] -to [get_pins {qubichw_config/jesdreset0xdomain/*/D}] 16.0
