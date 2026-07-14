`timescale 1ns / 100ps
module qubicdsp_tb(
);
regmap #(.LBCWIDTH(8),.LBAWIDTH(24),.LBDWIDTH(32)) lbreg();
dsp dsp();
qubicdsp qubicdsp(.dsp(dsp.dsp),.lbreg(lbreg.dsp));

reg sysclk=0;
integer cc=0;
initial begin
	$dumpfile("qubicdsp.vcd");
	$dumpvars(17,qubicdsp_tb);
	while (1) begin
		cc=cc+1;
		sysclk=0; #2.5;
		sysclk=1; #2.5;
	end
	$finish();
end

reg [31:0] sysclkcnt=0;
always @(posedge sysclk) begin
	sysclkcnt<=sysclkcnt+1;
end

localparam LBCWIDTH=8;
localparam LBAWIDTH=24;
localparam LBDWIDTH=32;
reg lb_clk=0;
initial begin
    forever #(4) lb_clk=~lb_clk;
end
reg [31:0] lbclkcnt=0;
always @(posedge lb_clk) begin
	lbclkcnt<=lbclkcnt+1;
end

reg dspclk=0;
initial begin
    forever #(2) dspclk=~dspclk;
end
assign qubicdsp.dsp.clk=dspclk;
reg [31:0] dspclkcnt=0;

always @(posedge qubicdsp.dsp.clk) begin
   dspclkcnt<=dspclkcnt+1;
end

assign qubicdsp.dsp.adc0=qubicdsp.dsp.dac0;
assign qubicdsp.dsp.adc1=qubicdsp.dsp.dac1;
assign qubicdsp.dsp.adc2=qubicdsp.dsp.dac2;
assign qubicdsp.dsp.adc3=qubicdsp.dsp.dac3;
assign qubicdsp.dsp.adc4=qubicdsp.dsp.dac4;
assign qubicdsp.dsp.adc5=qubicdsp.dsp.dac5;
assign qubicdsp.dsp.adc6=qubicdsp.dsp.dac6;
assign qubicdsp.dsp.adc7=qubicdsp.dsp.dac7;

reg lb_wvalid=0;
reg lb_read=0;
reg [LBCWIDTH-1:0] lb_wctrl=0;
reg [LBAWIDTH-1:0] lb_addr=0;
reg [LBDWIDTH-1:0] lb_wdata=0;

`include "simcmdin.vh"

assign lbreg.lb.clk=lb_clk;
assign lbreg.lb.waddr=lb_addr;
assign lbreg.lb.wdata=lb_wdata;
assign lbreg.lb.wvalid=lb_wvalid;
assign lbreg.lb.wctrl=lb_wctrl;
assign lbreg.lb.writecmd=8'h01;
assign lbreg.lb.read=lb_read;

endmodule
