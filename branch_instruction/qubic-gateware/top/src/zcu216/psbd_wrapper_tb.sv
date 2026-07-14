//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2022.1 (lin64) Build 3526262 Mon Apr 18 15:47:01 MDT 2022
//Date        : Sun Oct 30 12:43:08 2022
//Host        : new-begcbp-desktop running 64-bit Ubuntu 22.04.1 LTS
//Command     : generate_target psbd_wrapper.bd
//Design      : psbd_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1ns / 10ps

module psbd_wrapper_tb
(
);
localparam SIM=1;
reg clk100=0;
initial begin
	$dumpfile("psbd.vcd");
	$dumpvars(17,psbd_wrapper_tb);
while (1) begin
	clk100=0; #5;
	clk100=1; #5;
end
$finish();
end

reg clk125=0;
initial begin
	forever #(4) clk125=~clk125;
end
reg usersi570c0=0;
initial begin
	forever #(2.00) usersi570c0=~usersi570c0;
end
reg usersi570c1=0;
initial begin
	forever #(2.00) usersi570c1=~usersi570c1;
end
reg clk104_pl_sysref=0;
initial begin
	forever #(2.00) clk104_pl_sysref=~clk104_pl_sysref;
end
reg clk104_pl_clk=0;
initial begin
	forever #(2.00) clk104_pl_clk=~clk104_pl_clk;
end
reg adc2_clk=0;
initial begin
	forever #(1.000) adc2_clk=~adc2_clk;
end
reg dac3_clk=0;
initial begin
	forever #(1.000) dac3_clk=~dac3_clk;
end
reg dac2_clk=0;
initial begin
	forever #(1.000) dac2_clk=~dac2_clk;
end
reg sysref_in=0;
initial begin
	forever #(106.667) sysref_in=~sysref_in;
end

reg clk104_sync_in=0;

wire adc2_clk_clk_n=~adc2_clk;
wire adc2_clk_clk_p=adc2_clk;
wire dac2_clk_clk_n=~dac2_clk;
wire dac2_clk_clk_p=dac2_clk;
wire dac3_clk_clk_n=~dac3_clk;
wire dac3_clk_clk_p=dac3_clk;
wire sysref_in_diff_n=~sysref_in;
wire sysref_in_diff_p=sysref_in;

fpga fpga();
hwif hw();
assign hw.clk100=clk100;
assign hw.clk125=clk125;
assign hw.usersi570c0=usersi570c0;
assign hw.usersi570c1=usersi570c1;
assign hw.clk104_pl_sysref=clk104_pl_sysref;
assign hw.clk104_pl_clk=clk104_pl_clk;
board_sim board_sim(.fpga(fpga),.hw(hw.sim));

psbd_wrapper psbd_wrapper_i
(
//	.adc2_clk_clk_n(adc2_clk_clk_n),
//	.adc2_clk_clk_p(adc2_clk_clk_p),
	.dac2_clk_clk_n(dac2_clk_clk_n),
	.dac2_clk_clk_p(dac2_clk_clk_p),
	.fpga_a12(fpga.A12),
	.fpga_a13(fpga.A13),
	.fpga_an14(fpga.AN14),
	.fpga_ap14(fpga.AP14),
	.fpga_ap16(fpga.AP16),
	.fpga_ap21(fpga.AP21),
	.fpga_ar19(fpga.AR19),
	.fpga_ar20(fpga.AR20),
	.fpga_ar21(fpga.AR21),
	.fpga_au16(fpga.AU16),
	.fpga_av18(fpga.AV18),
	.fpga_av21(fpga.AV21),
	.fpga_aw12(fpga.AW12),
	.fpga_aw18(fpga.AW18),
	.fpga_ay10(fpga.AY10),
	.fpga_ay11(fpga.AY11),
	.fpga_ay16(fpga.AY16),
	.fpga_ay9(fpga.AY9),
	.fpga_b26(fpga.B26),
	.fpga_ba10(fpga.BA10),
	.fpga_ba19(fpga.BA19),
	.fpga_ba9(fpga.BA9),
	.fpga_bb10(fpga.BB10),
	.fpga_bb11(fpga.BB11),
	.fpga_bb12(fpga.BB12),
	.fpga_bb9(fpga.BB9),
	.fpga_c13(fpga.C13),
	.fpga_d11(fpga.D11),
	.fpga_d12(fpga.D12),
	.fpga_d13(fpga.D13),
	.fpga_d14(fpga.D14),
	.fpga_e10(fpga.E10),
	.fpga_e11(fpga.E11),
	.fpga_e24(fpga.E24),
	.fpga_e25(fpga.E25),
	.fpga_e9(fpga.E9),
	.fpga_f17(fpga.F17),
	.fpga_g11(fpga.G11),
	.fpga_g12(fpga.G12),
	.fpga_g13(fpga.G13),
	.fpga_g16(fpga.G16),
	.fpga_g17(fpga.G17),
	.fpga_g26(fpga.G26),
	.fpga_h10(fpga.H10),
	.fpga_h13(fpga.H13),
	.fpga_h14(fpga.H14),
	.fpga_h15(fpga.H15),
	.fpga_j11(fpga.J11),
	.fpga_j12(fpga.J12),
	.fpga_j13(fpga.J13),
	.fpga_j14(fpga.J14),
	.fpga_j23(fpga.J23),
	.fpga_k11(fpga.K11),
	.fpga_k12(fpga.K12),
	.fpga_l15(fpga.L15),
	.fpga_l17(fpga.L17),
	.fpga_l24(fpga.L24),
	.fpga_m14(fpga.M14),
	.fpga_m15(fpga.M15),
	.fpga_m16(fpga.M16),
	.fpga_m17(fpga.M17),
	.fpga_n14(fpga.N14),
	.fpga_n15(fpga.N15),
	.fpga_n16(fpga.N16),
	.fpga_p21(fpga.P21),
	.sysref_in_diff_n(sysref_in_diff_n),
	.sysref_in_diff_p(sysref_in_diff_p),
	.vout20_v_n(vout20_v_n),
	.vout20_v_p(vout20_v_p),
	.vout30_v_n(vout30_v_n),
	.vout30_v_p(vout30_v_p));
endmodule
