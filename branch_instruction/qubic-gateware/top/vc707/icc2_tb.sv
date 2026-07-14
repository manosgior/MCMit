//`include "constants.vams"
//`include "xc7vx485tffg1761pkg.vh"
`timescale 1ns / 1fs
module icc2_tb(
);

reg sysclk=0;
initial begin
	$dumpfile("icc2.vcd");
	$dumpvars(17,icc2_tb);
    forever #(2.5) sysclk=~sysclk;
	/*while (1) begin
		sysclk=0; #2.5;
		sysclk=1; #2.5;
	end*/
	$finish();
end
localparam SIM=1;
localparam DWIDTH=20;
reg [31:0] sysclkcnt=0;
always @(posedge sysclk) begin
	sysclkcnt<=sysclkcnt+1;
end
reg sma_mgt_refclk2=0;
reg dspclk=0;
initial begin
    forever #(2) dspclk=~dspclk;
end
reg [63:0] dspclk_cnt=0;
always @(posedge dspclk) begin
    dspclk_cnt<=dspclk_cnt+1;
end
localparam REFDLY=0;//.3;
localparam REFDLY2=0;//.39;
localparam REFDLY3=0;//.16;
reg dspclk2=0;
always @(*) begin
	if (dspclk_cnt>4321)
		dspclk2=#(REFDLY) dspclk;
end
reg [63:0] dspclk2_cnt=0;
always @(posedge dspclk2) begin
	dspclk2_cnt<=dspclk2_cnt+1;
end

always @(*) begin
	sma_mgt_refclk2=#(REFDLY3) dspclk2_cnt[0];
end
reg [2:0] smamgtclk_cnt2=0;
always @(posedge sma_mgt_refclk2) begin
    smamgtclk_cnt2<=smamgtclk_cnt2+1;
end

wire rxcore_p,rxcore_n;
wire txcore_p,txcore_n;
wire rxnocore_p,rxnocore_n;
wire txnocore_p,txnocore_n;
assign {rxnocore_p,rxnocore_n}=
	{txnocore_p,txnocore_n};
	//{txcore_p,txcore_n};
assign {rxcore_p,rxcore_n}=
	{txnocore_p,txnocore_n};
	//{txnocore_p,txnocore_n};
	//{txcore_p,txcore_n};
/*reg txnocore_pd=0,txnocore_nd=0;
always @(*) begin
	{txnocore_pd,txnocore_nd}=#(4280){txnocore_p,txnocore_n};
end
*/
//iicc #(.DWIDTH(16),.SIM(SIM)) iicc_nocore();
//iicc #(.DWIDTH(16),.SIM(SIM)) iicc_usecore();

igticc #(.DWIDTH(16)) igticc_nocore();
igticc #(.DWIDTH(16)) igticc_usecore();

gt_1G_125M_62_5M#(.SIM_CPLLREFCLK_SEL(3'h002),.DWIDTH(16),.SIM(SIM))
gt_usecore(.CPLLLOCKDETCLK(sysclk)
,.GTNORTHREFCLK0(1'b0),.GTNORTHREFCLK1(1'b0),.GTREFCLK0(1'b0),.GTREFCLK1(sma_mgt_refclk2),.GTSOUTHREFCLK0(1'b0),.GTSOUTHREFCLK1(1'b0)
,.GTXRXN(rxcore_n),.GTXRXP(rxcore_p),.GTXTXN(txcore_n),.GTXTXP(txcore_p)
,.QPLLCLK(1'b0),.QPLLREFCLK(1'b0)
,.CPLLREFCLKSEL(3'h2)
//,.stb_txphase(0)
//,.txphase(0)
//,.sfplos(sfplos)
//,.txphtarget(1)
//,.bypasstxphcheck(1'b1)
,.gticc(igticc_usecore.gt)
);

gt_1G_125M_no8b10b#(.SIM_CPLLREFCLK_SEL(3'h2),.DWIDTH(DWIDTH),.SIM(SIM))
gt_no8b10b(.CPLLLOCKDETCLK(sysclk)
,.GTNORTHREFCLK0(1'b0),.GTNORTHREFCLK1(1'b0),.GTREFCLK0(1'b0),.GTREFCLK1(sma_mgt_refclk2),.GTSOUTHREFCLK0(1'b0),.GTSOUTHREFCLK1(1'b0)
,.GTXRXN(rxnocore_n),.GTXRXP(rxnocore_p),.GTXTXN(txnocore_n),.GTXTXP(txnocore_p)
,.QPLLCLK(1'b0),.QPLLREFCLK(1'b0)
,.CPLLREFCLKSEL(3'h2)
,.gticc(igticc_nocore.gt)
//,.sfplos(1'b0)
//,.stb_txphase(0)
//,.bypasstxphcheck(1'b0)
//,.txphase(0)
//,.txphtarget(5'd19)
);

wire [15:0] txdata;
wire [1:0] txcharisk;
wire reset=sysclkcnt==4000;
wire reset_sfp_w2;
areset resetsfpareset2(.clk(sysclk),.areset(reset),.sreset(reset_sfp_w2));
assign igticc_nocore.reset=(reset_sfp_w2);
assign igticc_nocore.txdata=txdata;//16'h00bc;//7878;
assign igticc_nocore.txcharisk=txcharisk;//2'b01;//txcnt[5:1]==0 ? 2'b11 : 0;
//assign iicc_nocore.stbtxdata=1'b1;
//assign sfptestrx2=iicc_nocore.rxdata;
//assign iicc_nocore.master=1;
assign igticc_nocore.rxuserrdy=1'b1;
assign igticc_nocore.txuserrdy=1'b1;

//assign iicc_nocore.txcnt=dspclk2_cnt[49:2];
//assign slave.rxcnt=slave.usecorr ? dspclkcntcorr[49:2] : dspclk2_cnt[49:2];
assign igticc_usecore.reset=(reset_sfp_w2);
assign igticc_usecore.txdata=txdata;//16'h00bc;
assign igticc_usecore.txcharisk=txcharisk;//2'b01;//txcnt[5:1]==0 ? 2'b11 : 0;
//assign iicc_usecore.stbtxdata=1'b1;
//assign iicc_usecore.master=1;
assign igticc_usecore.rxuserrdy=1'b1;
assign igticc_usecore.txuserrdy=1'b1;
reg [31:0] txcnt=0;
//always @( posedge iicc_nocore.txclk) begin
always @( posedge igticc_usecore.txusrclk) begin
	txcnt<=txcnt+1;
end
//assign txdata=txcnt[5:1]==0 ? 16'h5cbc : txcnt[16:1];//{8'hf0,txcnt[8:1]};
assign txdata=txcnt[5:1]==0 ? 16'h1cbc : 16'h0000;//txcnt[16:1];//{8'hf0,txcnt[8:1]};
assign txcharisk=txcnt[5:1]==0 ? 2'b11 : 2'b00;//{8'hf0,txcnt[8:1]};
//assign txdata=16'hbcbc;//txcnt[5:1]==0 ? 16'h5cbc : 16'h00bc;//txcnt[16:1];//{8'hf0,txcnt[8:1]};
//assign txcharisk=2'b11;//txcnt[5:1]==0 ? 2'b11 : 2'b01;//{8'hf0,txcnt[8:1]};

endmodule
