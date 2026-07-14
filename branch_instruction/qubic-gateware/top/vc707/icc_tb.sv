//`include "constants.vams"
//`include "xc7vx485tffg1761pkg.vh"
`timescale 1ns / 1fs
module icc_tb(
);

reg sysclk=0;
initial begin
	$dumpfile("icc.vcd");
	$dumpvars(17,icc_tb);
    forever #(2.5) sysclk=~sysclk;
	/*while (1) begin
		sysclk=0; #2.5;
		sysclk=1; #2.5;
	end*/
	$finish();
end
localparam SIM=1;
localparam DWIDTH=16;
reg [31:0] sysclkcnt=0;
always @(posedge sysclk) begin
	sysclkcnt<=sysclkcnt+1;
end
reg dspclk=0;
initial begin
    forever #(2) dspclk=~dspclk;
end
reg [63:0] dspclk_cnt=0;
always @(posedge dspclk) begin
    dspclk_cnt<=dspclk_cnt+1;
end
localparam REFDLY=0.3;
localparam REFDLY2=0.39;
localparam REFDLY3=0.16;
reg dspclk2=0;
always @(*) begin
	if (dspclk_cnt>4321)
		dspclk2=#(REFDLY) dspclk;
end
reg [63:0] dspclk2_cnt=0;
always @(posedge dspclk2) begin
	dspclk2_cnt<=dspclk2_cnt+1;
end
reg sma_mgt_refclk=0;
always @(*) begin
	sma_mgt_refclk= #REFDLY2 dspclk_cnt[0];
end
//initial begin
 //   forever #(4) sma_mgt_refclk=~sma_mgt_refclk;
//end
reg helpclk=0;
initial begin
    forever #(8.010488) helpclk=~helpclk;
end

reg [2:0] smamgtclk_cnt=0;
always @(posedge sma_mgt_refclk) begin
    smamgtclk_cnt<=smamgtclk_cnt+1;
end
reg sma_mgt_refclk2=0;
always @(*) begin
	sma_mgt_refclk2=#(REFDLY3) dspclk2_cnt[0];
end
reg [2:0] smamgtclk_cnt2=0;
always @(posedge sma_mgt_refclk2) begin
    smamgtclk_cnt2<=smamgtclk_cnt2+1;
end

wire reset_sfp=sysclkcnt==1500;
wire reset_sfp2=sysclkcnt==2000;
wire [DWIDTH-1:0] sfptestrx;
wire sfpreconnected=1'b0;
wire sfp_tx_p,sfp_tx_n,sfp_rx_p,sfp_rx_n;
wire sfp2_tx_p,sfp2_tx_n,sfp2_rx_p,sfp2_rx_n;

reg sfptxpr=0,sfptxnr=0,sfp2txpr=0,sfp2txnr=0;
reg sfptxpr_1=0,sfptxnr_1=0,sfp2txpr_1=0,sfp2txnr_1=0;
reg sfptxpr_2=0,sfptxnr_2=0,sfp2txpr_2=0,sfp2txnr_2=0;
reg sfptxpr_3=0,sfptxnr_3=0,sfp2txpr_3=0,sfp2txnr_3=0;
reg sfptxpr_4=0,sfptxnr_4=0,sfp2txpr_4=0,sfp2txnr_4=0;
reg sfptxpr_5=0,sfptxnr_5=0,sfp2txpr_5=0,sfp2txnr_5=0;
reg sfptxpr_6=0,sfptxnr_6=0,sfp2txpr_6=0,sfp2txnr_6=0;
reg sfptxpr_7=0,sfptxnr_7=0,sfp2txpr_7=0,sfp2txnr_7=0;
reg sfptxpr_8=0,sfptxnr_8=0,sfp2txpr_8=0,sfp2txnr_8=0;
localparam D1=0;
localparam D2=0;
always @(*) sfptxpr= #D1 sfp_tx_p;
always @(*) sfptxnr= #D1 sfp_tx_n;
always @(*) sfp2txpr=#D1 sfp2_tx_p;
always @(*) sfp2txnr=#D1 sfp2_tx_n;
always @(*) sfptxpr_1= #D2 sfptxpr;
always @(*) sfptxnr_1= #D2 sfptxnr;
always @(*) sfp2txpr_1=#D2 sfp2txpr;
always @(*) sfp2txnr_1=#D2 sfp2txnr;
always @(*) sfptxpr_2= #D2 sfptxpr_1;
always @(*) sfptxnr_2= #D2 sfptxnr_1;
always @(*) sfp2txpr_2=#D2 sfp2txpr_1;
always @(*) sfp2txnr_2=#D2 sfp2txnr_1;
always @(*) sfptxpr_3= #D2 sfptxpr_2;
always @(*) sfptxnr_3= #D2 sfptxnr_2;
always @(*) sfp2txpr_3=#D2 sfp2txpr_2;
always @(*) sfp2txnr_3=#D2 sfp2txnr_2;
always @(*) sfptxpr_4= #D2 sfptxpr_3;
always @(*) sfptxnr_4= #D2 sfptxnr_3;
always @(*) sfp2txpr_4=#D2 sfp2txpr_3;
always @(*) sfp2txnr_4=#D2 sfp2txnr_3;
always @(*) sfptxpr_5= #D2 sfptxpr_4;
always @(*) sfptxnr_5= #D2 sfptxnr_4;
always @(*) sfp2txpr_5=#D2 sfp2txpr_4;
always @(*) sfp2txnr_5=#D2 sfp2txnr_4;
always @(*) sfptxpr_6= #D2 sfptxpr_5;
always @(*) sfptxnr_6= #D2 sfptxnr_5;
always @(*) sfp2txpr_6=#D2 sfp2txpr_5;
always @(*) sfp2txnr_6=#D2 sfp2txnr_5;
always @(*) sfptxpr_7= #D2 sfptxpr_6;
always @(*) sfptxnr_7= #D2 sfptxnr_6;
always @(*) sfp2txpr_7=#D2 sfp2txpr_6;
always @(*) sfp2txnr_7=#D2 sfp2txnr_6;
always @(*) sfptxpr_8= #D2 sfptxpr_7;
always @(*) sfptxnr_8= #D2 sfptxnr_7;
always @(*) sfp2txpr_8=#D2 sfp2txpr_7;
always @(*) sfp2txnr_8=#D2 sfp2txnr_7;

assign {sfp2_rx_p,sfp2_rx_n} ={sfptxpr_8,sfptxnr_8};//{sfp_tx_p,sfp_tx_n};//
//{sfptxpr,sfptxnr};//
assign {sfp_rx_p,sfp_rx_n}={sfp2txpr_8,sfp2txnr_8};//{sfp2_tx_p,sfp2_tx_n};
// {sfp2txpr,sfp2txnr};//

wire stb_txphaseab_x;
wire stb_txphaseab;
wire [31:0] txphaseab;
iicc #(.DWIDTH(16),.SIM(SIM)) sfpiccmaster();
gticc_gt #(.SIM_CPLLREFCLK_SEL(3'h2),.DWIDTH(DWIDTH),.TXPHTARGET(5'h0),.SIM(SIM))
gticc_gt_sfp(.CPLLLOCKDETCLK(sysclk)
,.GTNORTHREFCLK0(1'b0),.GTNORTHREFCLK1(1'b0),.GTREFCLK0(1'b0),.GTREFCLK1(sma_mgt_refclk),.GTSOUTHREFCLK0(1'b0),.GTSOUTHREFCLK1(1'b0)
,.GTXRXN(sfp_rx_n),.GTXRXP(sfp_rx_p),.GTXTXN(sfp_tx_n),.GTXTXP(sfp_tx_p)
,.QPLLCLK(1'b0),.QPLLREFCLK(1'b0)
,.CPLLREFCLKSEL(3'h2)
,.gticc(sfpiccmaster.gticc.gt)
,.stb_txphase(stb_txphaseab_x)
,.txphase(txphaseab)
,.txphtarget(5'd0)
,.bypasstxphcheck(1'b0)
,.mask(6'h3f)
);
wire reset_sfp_w;
areset resetsfpareset(.clk(sysclk),.areset(reset_sfp),.sreset(reset_sfp_w));
assign sfpiccmaster.sreset=(reset_sfp_w|sfpreconnected);
assign sfpiccmaster.txdata=16'h5c5c;
assign sfpiccmaster.stbtxdata=1'b1;
assign sfptestrx=sfpiccmaster.rxdata;
assign sfpiccmaster.master=1;
assign sfpiccmaster.txcnt=dspclk_cnt[49:2];
assign sfpiccmaster.rxcnt=dspclk_cnt[49:2];
wire [31:0] txphaseab2;
wire stb_txphaseab_x2;
wire [DWIDTH-1:0] sfptestrx2;
wire sfpreconnected2=1'b0;
iicc #(.DWIDTH(16),.SIM(SIM)) sfpiccslave();
gticc_gt #(.SIM_CPLLREFCLK_SEL(3'h2),.DWIDTH(DWIDTH),.TXPHTARGET(5'd19),.SIM(SIM))
gticc_gt_sfp2(.CPLLLOCKDETCLK(sysclk)
,.GTNORTHREFCLK0(1'b0),.GTNORTHREFCLK1(1'b0),.GTREFCLK0(1'b0),.GTREFCLK1(sma_mgt_refclk2),.GTSOUTHREFCLK0(1'b0),.GTSOUTHREFCLK1(1'b0)
,.GTXRXN(sfp2_rx_n),.GTXRXP(sfp2_rx_p),.GTXTXN(sfp2_tx_n),.GTXTXP(sfp2_tx_p)
,.QPLLCLK(1'b0),.QPLLREFCLK(1'b0)
,.CPLLREFCLKSEL(3'h2)
,.gticc(sfpiccslave.gticc.gt)
,.mask(6'h3f)
,.stb_txphase(stb_txphaseab_x2)
,.bypasstxphcheck(1'b0)
,.txphase(txphaseab2)
,.txphtarget(5'd19)
);
wire reset_sfp_w2;
areset resetsfpareset2(.clk(sysclk),.areset(reset_sfp2),.sreset(reset_sfp_w2));
assign sfpiccslave.sreset=(reset_sfp_w2|sfpreconnected2);
assign sfpiccslave.txdata=16'h7878;
assign sfpiccslave.stbtxdata=1'b1;
assign sfptestrx2=sfpiccslave.rxdata;
assign sfpiccslave.master=0;

wire [63:0] corr48_ext64;
wire [63:0] corr4_ext64;
sext #(.WIN(2),.WOUT(64)) corr4ext64 (.din(sfpiccslave.corr4>>>2),.dout(corr4_ext64));
sext #(.WIN(48),.WOUT(64)) corr48ext64(.din(sfpiccslave.corr48<<<2),.dout(corr48_ext64));
wire [63:0] corr64=corr48_ext64-corr4_ext64;
wire [63:0] dspclk2cntcorr=dspclk2_cnt+corr64;
assign sfpiccslave.txcnt=sfpiccslave.usecorr ? dspclk2cntcorr[49:2] : dspclk2_cnt[49:2];
//assign sfpiccslave.rxcnt=sfpiccslave.usecorr ? dspclkcntcorr[49:2] : dspclk2_cnt[49:2];

wire dmtdreset_w=0;
wire [31:0] stableval=200;
wire [31:0] rxphaseab;
wire stb_rxphaseab;
wire [31:0] txphaseab_x;
wire [31:0] rxphaseab_x;
wire stb_rxphaseab_x;
dmtd #(.SIM(SIM),.FOFFSETWIDTH(SIM ? 9 : 14))
dmtdmasterrx (.clkdmtd(helpclk),.rst(dmtdreset_w),.stableval(stableval)
,.clka(dspclk_cnt[1]),.clkb(sfpiccmaster.gticc.rxusrclk),.phaseab(rxphaseab),.stb_phaseab(stb_rxphaseab)
);
dmtd #(.SIM(SIM),.FOFFSETWIDTH(SIM ? 9 : 14))
dmtdmastertx (.clkdmtd(helpclk),.rst(dmtdreset_w),.stableval(stableval)
,.clka(dspclk_cnt[1]),.clkb(sfpiccmaster.gticc.txusrclk),.phaseab(txphaseab),.stb_phaseab(stb_txphaseab)
);

data_xdomain #(.DWIDTH(32)) phaselatchmastertx(.clkin(helpclk),.clkout(sfpiccmaster.gticc.txusrclk),.gatein(stb_txphaseab),.datain(txphaseab),.gateout(stb_txphaseab_x),.dataout(txphaseab_x));
data_xdomain #(.DWIDTH(32)) phaselatchmasterrx(.clkin(helpclk),.clkout(sfpiccmaster.gticc.txusrclk),.gatein(stb_rxphaseab),.datain(rxphaseab),.gateout(stb_rxphaseab_x),.dataout(rxphaseab_x));
assign sfpiccmaster.txphdmtd=txphaseab_x[15:0];
assign sfpiccmaster.rxphdmtd=rxphaseab_x[15:0];


wire dmtdreset_w2=0;
wire [31:0] stableval2=200;
wire stb_txphaseab2;
wire [31:0] txphaseab_x2;
wire [31:0] rxphaseab2;
wire stb_rxphaseab2;
wire [31:0] rxphaseab_x2;
wire stb_rxphaseab_x2;
dmtd #(.SIM(SIM),.FOFFSETWIDTH(SIM ? 9 : 14))
dmtdslavetx(.clkdmtd(helpclk),.rst(dmtdreset_w2),.stableval(stableval2)
,.clka(smamgtclk_cnt2[0]),.clkb(sfpiccslave.gticc.txusrclk)
,.phaseab(txphaseab2),.stb_phaseab(stb_txphaseab2)
);
dmtd #(.SIM(SIM),.FOFFSETWIDTH(SIM ? 9 : 14))
dmtdslaverx(.clkdmtd(helpclk),.rst(dmtdreset_w2),.stableval(stableval2)
,.clka(smamgtclk_cnt2[0]),.clkb(sfpiccslave.gticc.rxusrclk)
,.phaseab(rxphaseab2),.stb_phaseab(stb_rxphaseab2)
);

data_xdomain #(.DWIDTH(32)) phaselatchslaverx(.clkin(helpclk),.clkout(sfpiccslave.gticc.txusrclk),.gatein(stb_rxphaseab2),.datain(rxphaseab2),.gateout(stb_rxphaseab_x2),.dataout(rxphaseab_x2));
data_xdomain #(.DWIDTH(32)) phaselatchslavetx(.clkin(helpclk),.clkout(sfpiccslave.gticc.txusrclk),.gatein(stb_txphaseab2),.datain(txphaseab2),.gateout(stb_txphaseab_x2),.dataout(txphaseab_x2));
assign sfpiccslave.rxphdmtd=rxphaseab_x2[15:0];
assign sfpiccslave.txphdmtd=txphaseab_x2[15:0];
endmodule
