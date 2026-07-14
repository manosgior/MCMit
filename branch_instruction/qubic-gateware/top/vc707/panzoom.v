`timescale 1ns / 1ns

module panzoom(clk, trig, reset, din, opsel, dout, navr, dt, gout, addrcnt, stopped);
parameter DW=16;
parameter MAXDAVR=20;
parameter NCHAN=2;
parameter MEMAW=10;
input clk;
input trig;
input [5:0] navr;
input [31:0] dt;
input reset;
input [NCHAN*DW-1:0] din;
input [2:0] opsel;
output [NCHAN*DW-1:0] dout;
output gout;
output [MEMAW-1:0] addrcnt;
output stopped;
wire stop;
reg signed [DW+$clog2(MAXDAVR)-1:0] inte [NCHAN-1:0];
reg signed [DW-1:0] dmin [NCHAN-1:0];
reg signed [DW-1:0] dmax [NCHAN-1:0];
integer inichan = 0;
initial begin
	for (inichan=0;inichan<NCHAN;inichan=inichan+1) begin
		inte[inichan] = 0;
		dmin[inichan] = {1'b0,{(DW-1){1'b1}}};
		dmax[inichan] = {1'b1,{(DW-1){1'b0}}};
	end
end
reg [MEMAW:0] addrcnt_r = 0;
reg [MEMAW:0] addrcnt_d = 0;
reg [MEMAW:0] addrcnt_d2 = 0;
reg [31:0] ncnt = 0;
reg [31:0] navrpw = 0;
reg [31:0] tcnt = 0;
reg [5:0] navr_d = 0;
wire [NCHAN*DW-1:0] dout_w;
wire nochange = (navr==navr_d);
wire changed = !nochange;
wire ncntlast = (ncnt==(navrpw-1));
reg start = 0;
assign stop = addrcnt_r[MEMAW];
reg reseted = 0;
reg reset_d = 0;
reg started = 0;
reg started_d = 0;
assign stopped=~started_d;
wire firststart = started & ~started_d;
reg ncntlast_d = 0;
reg ncntlast_d2 = 0;

always @(posedge clk) begin
	navrpw <= 1<<navr;
	navr_d <= navr;
	reset_d <= reset;
	if (reset&~reset_d)
		reseted <= 1'b1;
	else if (stop)
		reseted <= 1'b0;
	if (reseted&trig)
		started <= 1'b1;
	else if (stop)
		started <= 0;
	started_d <= started;

	navrpw <= 1<<navr;
	tcnt <= ~started ? 0 : tcnt +1;
	ncnt <= firststart|ncntlast|(tcnt<dt)|changed|stop ? 0 : ncnt +1;
	//addrcnt_r <= reset|changed ? 0 : addrcnt_r + (ncntlast & started);
	addrcnt_r <= (navr==0) ? (reset|changed|stop ? 0 : addrcnt_r + ((tcnt>=dt) & started)) : (reset|changed|stop ? 0 : addrcnt_r + (ncntlast & started));
	addrcnt_d <= addrcnt_r;
	addrcnt_d2 <= addrcnt_d;
	ncntlast_d <= ncntlast;
	ncntlast_d2 <= ncntlast_d;
end

//reg [31:0] test_cnt=0;
//always @(posedge clk) begin
//	test_cnt <= test_cnt+1;
//end

genvar ichan;
generate for (ichan=0;ichan<NCHAN; ichan=ichan+1) begin: chanloop
	wire signed [DW-1:0] dini = din[(ichan+1)*DW-1:(ichan*DW)];  //test_cnt[(ichan+1)*DW-1:(ichan*DW)];
	reg signed [DW-1:0] douti = 0;
	wire [DW+$clog2(MAXDAVR)-1:0] intemon;
	wire [DW-1:0] dminmon, dmaxmon;
	always @(posedge clk) begin
		if (~|ncnt) begin
			douti <= (opsel==3'b000) ? (inte[ichan]>>>navr) : (opsel==3'b001) ? dmin[ichan] : (opsel==3'b010) ? dmax[ichan] : 16'b0;
			inte[ichan] <= dini;
			dmin[ichan] <= dini;
			dmax[ichan] <= dini;
		end
		else begin
			inte[ichan] <= inte[ichan] + dini;
			dmin[ichan] <= (dini<dmin[ichan]) ? dini : dmin[ichan];
			dmax[ichan] <= (dini>dmax[ichan]) ? dini : dmax[ichan];
		end
	end
	assign dout_w[(ichan+1)*DW-1:ichan*DW] = douti;
	assign intemon = inte[ichan];
	assign dminmon = dmin[ichan];
	assign dmaxmon = dmax[ichan];
end
endgenerate

assign dout = dout_w;
assign gout = ncntlast_d2 & started_d;
//assign gout = ncntlast_d2;
assign addrcnt = addrcnt_d2[MEMAW-1:0];

endmodule
