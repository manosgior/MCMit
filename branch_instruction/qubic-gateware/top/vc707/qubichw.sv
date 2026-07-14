module qubichw (hw hw,xc7vx485tffg1761pkg fpga);
vc707 vc707(.fpga(fpga),.hw(hw.vc707.hw));
fmc120 fmc120_fmc1(.fmcpin(hw.vc707.fmc1pin.pin),.fmc120(hw.fmc1.hw));
fmc120_2 fmc120_fmc2(.fmcpin(hw.vc707.fmc2pin.pin),.fmc120(hw.fmc2.hw));
endmodule

interface hw ();

//`include "vc707.vh"
ivc707 vc707();
ifmc120 fmc1();
ifmc120_2 fmc2();

/*modport hw(inout fmc_vadj_on_b_ls,phy_mdio,pmbus_alert,pmbus_clk,pmbus_data,user_clock_n,user_clock_p,user_sma_clock_n,user_sma_clock_p,user_sma_gpio_n,user_sma_gpio_p
,input gpio_led_0,gpio_led_1,gpio_led_2,gpio_led_3,gpio_led_4,gpio_led_5,gpio_led_6,gpio_led_7,phy_int,phy_mdc,phy_reset,sgmii_tx_n,sgmii_tx_p,si5324_rst,sm_fan_pwm
,output cpu_reset,fmc1_hpc_gbtclk0_m2c_c_n,fmc1_hpc_gbtclk0_m2c_c_p,fmc1_hpc_gbtclk1_m2c_c_n,fmc1_hpc_gbtclk1_m2c_c_p,fmc2_hpc_gbtclk0_m2c_c_n,fmc2_hpc_gbtclk0_m2c_c_p,fmc2_hpc_gbtclk1_m2c_c_n,fmc2_hpc_gbtclk1_m2c_c_p,gpio_dip_sw0,gpio_dip_sw1,gpio_dip_sw2,gpio_dip_sw3,gpio_dip_sw4,gpio_dip_sw5,gpio_dip_sw6,gpio_dip_sw7,gpio_sw_c,gpio_sw_e,gpio_sw_n,gpio_sw_s,gpio_sw_w,rec_clock_c_n,rec_clock_c_p,rotary_inca,rotary_incb,rotary_push,sgmii_rx_n,sgmii_rx_p,sgmiiclk_q0_n,sgmiiclk_q0_p,si5324_int_alm,si5324_out_c_n,si5324_out_c_p,sm_fan_tach,sma_mgt_refclk_n,sma_mgt_refclk_p,sma_mgt_rx_n,sma_mgt_rx_p,sma_mgt_tx_n,sma_mgt_tx_p,sysclk,VP_0,VN_0
);
modport cfg(inout phy_mdio,user_clock_n,user_clock_p,user_sma_clock_n,user_sma_clock_p,user_sma_gpio_n,user_sma_gpio_p
,input cpu_reset,gpio_dip_sw0,gpio_dip_sw1,gpio_dip_sw2,gpio_dip_sw3,gpio_dip_sw4,gpio_dip_sw5,gpio_dip_sw6,gpio_dip_sw7,gpio_sw_c,gpio_sw_e,gpio_sw_n,gpio_sw_s,gpio_sw_w,rotary_inca,rotary_incb,rotary_push,sgmii_rx_n,sgmii_rx_p,sgmiiclk_q0_n,sgmiiclk_q0_p,sysclk,VP_0,VN_0
,output gpio_led_0,gpio_led_1,gpio_led_2,gpio_led_3,gpio_led_4,gpio_led_5,gpio_led_6,gpio_led_7,phy_int,phy_mdc,phy_reset,sgmii_tx_n,sgmii_tx_p
);
modport sim(
inout fmc_vadj_on_b_ls,phy_mdio,pmbus_alert,pmbus_clk,pmbus_data,user_clock_n,user_clock_p,user_sma_clock_n,user_sma_clock_p,user_sma_gpio_n,user_sma_gpio_p
,output gpio_led_0,gpio_led_1,gpio_led_2,gpio_led_3,gpio_led_4,gpio_led_5,gpio_led_6,gpio_led_7,phy_int,phy_mdc,phy_reset,sgmii_tx_n,sgmii_tx_p,si5324_rst,sm_fan_pwm
,input cpu_reset,fmc1_hpc_gbtclk0_m2c_c_n,fmc1_hpc_gbtclk0_m2c_c_p,fmc1_hpc_gbtclk1_m2c_c_n,fmc1_hpc_gbtclk1_m2c_c_p,fmc2_hpc_gbtclk0_m2c_c_n,fmc2_hpc_gbtclk0_m2c_c_p,fmc2_hpc_gbtclk1_m2c_c_n,fmc2_hpc_gbtclk1_m2c_c_p,gpio_dip_sw0,gpio_dip_sw1,gpio_dip_sw2,gpio_dip_sw3,gpio_dip_sw4,gpio_dip_sw5,gpio_dip_sw6,gpio_dip_sw7,gpio_sw_c,gpio_sw_e,gpio_sw_n,gpio_sw_s,gpio_sw_w,rec_clock_c_n,rec_clock_c_p,rotary_inca,rotary_incb,rotary_push,sgmii_rx_n,sgmii_rx_p,sgmiiclk_q0_n,sgmiiclk_q0_p,si5324_int_alm,si5324_out_c_n,si5324_out_c_p,sm_fan_tach,sma_mgt_refclk_n,sma_mgt_refclk_p,sma_mgt_rx_n,sma_mgt_rx_p,sma_mgt_tx_n,sma_mgt_tx_p,sysclk,VP_0,VN_0
);
*/

/*wire fmc1__adcb_sync_in_l_vadj;
wire fmc1__adca_sync_in_l_vadj;
wire fmc2__adcb_sync_in_l_vadj;
wire fmc2__adca_sync_in_l_vadj;

wire fmc1__adc0_da1_p=fmc1.adc0_da1_p;
wire fmc1__adc0_da1_n=fmc1.adc0_da1_n;
wire fmc1__adc0_da2_p=fmc1.adc0_da2_p;
wire fmc1__adc0_da2_n=fmc1.adc0_da2_n;
wire fmc1__adc0_db1_p=fmc1.adc0_db1_p;
wire fmc1__adc0_db1_n=fmc1.adc0_db1_n;
wire fmc1__adc0_db2_p=fmc1.adc0_db2_p;
wire fmc1__adc0_db2_n=fmc1.adc0_db2_n;
wire fmc1__adc1_da1_p=fmc1.adc1_da1_p;
wire fmc1__adc1_da1_n=fmc1.adc1_da1_n;
wire fmc1__adc1_da2_p=fmc1.adc1_da2_p;
wire fmc1__adc1_da2_n=fmc1.adc1_da2_n;
wire fmc1__adc1_db1_p=fmc1.adc1_db1_p;
wire fmc1__adc1_db1_n=fmc1.adc1_db1_n;
wire fmc1__adc1_db2_p=fmc1.adc1_db2_p;
wire fmc1__adc1_db2_n=fmc1.adc1_db2_n;

wire fmc2__adc0_da1_p=fmc2.adc0_da1_p;
wire fmc2__adc0_da1_n=fmc2.adc0_da1_n;
wire fmc2__adc0_da2_p=fmc2.adc0_da2_p;
wire fmc2__adc0_da2_n=fmc2.adc0_da2_n;
wire fmc2__adc0_db1_p=fmc2.adc0_db1_p;
wire fmc2__adc0_db1_n=fmc2.adc0_db1_n;
wire fmc2__adc0_db2_p=fmc2.adc0_db2_p;
wire fmc2__adc0_db2_n=fmc2.adc0_db2_n;
wire fmc2__adc1_da1_p=fmc2.adc1_da1_p;
wire fmc2__adc1_da1_n=fmc2.adc1_da1_n;
wire fmc2__adc1_da2_p=fmc2.adc1_da2_p;
wire fmc2__adc1_da2_n=fmc2.adc1_da2_n;
wire fmc2__adc1_db1_p=fmc2.adc1_db1_p;
wire fmc2__adc1_db1_n=fmc2.adc1_db1_n;
wire fmc2__adc1_db2_p=fmc2.adc1_db2_p;
wire fmc2__adc1_db2_n=fmc2.adc1_db2_n;
assign fmc1.adcb_sync_in_l_vadj=fmc1__adcb_sync_in_l_vadj;
assign fmc1.adca_sync_in_l_vadj=fmc1__adca_sync_in_l_vadj;
assign fmc2.adcb_sync_in_l_vadj=fmc2__adcb_sync_in_l_vadj;
assign fmc2.adca_sync_in_l_vadj=fmc2__adca_sync_in_l_vadj;
wire [7:0] fmc1__dac_lane_p,fmc1__dac_lane_n;
wire [7:0] fmc2__dac_lane_p,fmc2__dac_lane_n;
assign fmc1.dac_lane_p=fmc1__dac_lane_p;
assign fmc1.dac_lane_n=fmc1__dac_lane_n;
assign fmc2.dac_lane_p=fmc2__dac_lane_p;
assign fmc2.dac_lane_n=fmc2__dac_lane_n;
modport cfg(output  fmc1__adcb_sync_in_l_vadj,fmc1__adca_sync_in_l_vadj,fmc2__adcb_sync_in_l_vadj,fmc2__adca_sync_in_l_vadj
,fmc1__dac_lane_p,fmc1__dac_lane_n,fmc2__dac_lane_p,fmc2__dac_lane_n
,input fmc1__adc0_da1_p,fmc1__adc0_da1_n,fmc1__adc0_da2_p,fmc1__adc0_da2_n,fmc1__adc0_db1_p,fmc1__adc0_db1_n,fmc1__adc0_db2_p,fmc1__adc0_db2_n,fmc1__adc1_da1_p,fmc1__adc1_da1_n,fmc1__adc1_da2_p,fmc1__adc1_da2_n,fmc1__adc1_db1_p,fmc1__adc1_db1_n,fmc1__adc1_db2_p,fmc1__adc1_db2_n
,fmc2__adc0_da1_p,fmc2__adc0_da1_n,fmc2__adc0_da2_p,fmc2__adc0_da2_n,fmc2__adc0_db1_p,fmc2__adc0_db1_n,fmc2__adc0_db2_p,fmc2__adc0_db2_n,fmc2__adc1_da1_p,fmc2__adc1_da1_n,fmc2__adc1_da2_p,fmc2__adc1_da2_n,fmc2__adc1_db1_p,fmc2__adc1_db1_n,fmc2__adc1_db2_p,fmc2__adc1_db2_n
);
modport hw(input fmc1__adcb_sync_in_l_vadj,fmc1__adca_sync_in_l_vadj,fmc2__adcb_sync_in_l_vadj,fmc2__adca_sync_in_l_vadj
);*/
endinterface
