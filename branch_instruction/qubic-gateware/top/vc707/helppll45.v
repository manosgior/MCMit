module helppll45#(parameter DWIDTH=32)(input clkref
,input clkhelp
,input [DWIDTH-1:0] refcntsamp
,output signed [DWIDTH-1:0] freqdiff
,output stb_freqdiff
,output [DWIDTH-1:0] dbclkhelpcnt_samp0
,output [DWIDTH-1:0] dbclkhelpcnt_samp1
,output [DWIDTH-1:0] dbfreqhelp
,output [DWIDTH-1:0] dbfreqhelp5
,output [DWIDTH-1:0] dbfreqref4
,output [DWIDTH-1:0] dbfreqdiff
,output dbsamp
,output dbsamphelp_d1_ref
,output [DWIDTH-1:0] dbclkrefcnt
,output dbsamphelp_d1_ref_v
,output dbsamphelp_v
);

reg [DWIDTH-1:0] clkrefcnt=0;
reg [DWIDTH+2:0] freqref4=0;
wire samp=(~|clkrefcnt);
reg samp_d=0;
always @(posedge clkref) begin
	samp_d<=samp;
	clkrefcnt<=samp_d ? (refcntsamp-1'b1) : (clkrefcnt-1'b1);
	freqref4<=refcntsamp;// 125-62_5 <<2;
//	freqref4<=refcntsamp <<2;
end
assign dbclkrefcnt=clkrefcnt;
wire samphelp;
reg samphelp_d=0;
reg samphelp_d1=0;
reg samphelp_d2=0;
reg samphelp_d3=0;
reg [DWIDTH+2:0] freqhelp=0;
reg [DWIDTH+2:0] freqhelp5=0;
areset aresetsamp(.clk(clkhelp),.areset(samp_d),.sreset(samphelp),.sreset_val(dbsamphelp_v));
reg [DWIDTH-1:0] clkhelpcnt=0;
reg [DWIDTH-1:0] clkhelpcnt_samp0=0;
reg [DWIDTH-1:0] clkhelpcnt_samp1=0;
always @(posedge clkhelp) begin
	clkhelpcnt<=clkhelpcnt+1;
	samphelp_d<=samphelp;
	samphelp_d1<=samphelp_d;
	samphelp_d2<=samphelp_d1;
	samphelp_d3<=samphelp_d2;
	if (samphelp) begin
		clkhelpcnt_samp0<=clkhelpcnt;
		clkhelpcnt_samp1<=clkhelpcnt_samp0;
	end
	freqhelp<=clkhelpcnt_samp0-clkhelpcnt_samp1;
	freqhelp5<=freqhelp;//<<1;// 125-62_5 (freqhelp<<2)+freqhelp;
//	freqhelp5<=(freqhelp<<2)+freqhelp;
end
reg [DWIDTH+2:0] freqhelp5_ref=0;
areset aresetsampd1(.clk(clkref),.areset(samphelp_d3),.sreset(samphelp_d1_ref),.sreset_val(dbsamphelp_d1_ref_v));
reg signed [DWIDTH-1:0] freqdiff_r=0;
reg signed [DWIDTH-1:0] dbfreqdiff_r=0;
reg stb_freqdiff_r=0;
reg stb_freqdiff_r2=0;
always @(posedge clkref) begin
	if (samphelp_d1_ref)
		freqhelp5_ref<=freqhelp5;
	freqdiff_r<=$signed(freqhelp5_ref)-$signed(freqref4);
	dbfreqdiff_r<=$signed(freqhelp5_ref)-$signed(freqref4);
	stb_freqdiff_r<=samphelp_d1_ref;
	stb_freqdiff_r2<=stb_freqdiff_r;
end
assign freqdiff=freqdiff_r;//>>>1;// 125-62_5 >>>2;
//assign freqdiff=freqdiff_r>>>2;
assign stb_freqdiff=stb_freqdiff_r2;

assign dbclkhelpcnt_samp0=clkhelpcnt_samp0;
assign dbclkhelpcnt_samp1=clkhelpcnt_samp1;
assign dbfreqhelp=freqhelp;
assign dbfreqhelp5=freqhelp5;
assign dbfreqref4=freqref4;
assign dbfreqdiff=dbfreqdiff_r;//>>>1;
assign dbsamp=samp_d;
assign dbsamphelp_d1_ref=samphelp_d1_ref;
endmodule
