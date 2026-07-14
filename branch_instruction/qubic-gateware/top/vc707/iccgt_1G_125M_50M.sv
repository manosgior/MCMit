module iccgt_1G_125M_50M#
(parameter DWIDTH=16
,parameter GT_SIM_GTRESET_SPEEDUP = "TRUE"
,parameter RX_DFE_KL_CFG2_IN = 32'h301148AC
,parameter SIM_CPLLREFCLK_SEL = 3'b001
,parameter PMA_RSV_IN = 32'h00018480
,parameter PCS_RSVD_ATTR_IN = 48'h000000000000
,localparam DBYTE=DWIDTH/8
,localparam NSTEP=6
,parameter SIM=0
)
(input GTXRXP
,input GTXRXN
,output GTXTXN
,output GTXTXP
,input GTNORTHREFCLK0
,input GTNORTHREFCLK1
,input GTREFCLK0
,input GTREFCLK1
,input GTSOUTHREFCLK0
,input GTSOUTHREFCLK1
,input QPLLCLK
,input QPLLREFCLK
,input CPLLLOCKDETCLK
,input [2:0] CPLLREFCLKSEL
,input stb_txphase
,input [15:0] txphase
,input [4:0] txphtarget
,input sfplos
,input phrefclk62_5
,input reset
,output txoutclk
,igticc.gt gticc
,input bypasstxphcheck
,output RXOUTCLKFABRIC
,output TXOUTCLKFABRIC
,output RXOUTCLKPCS
,output TXOUTCLKPCS
,output dblocked
,output dbrxcdrlock
,output [NSTEP-1:0] dbdonecriteria
,output  [24:0] dbdone
,output [NSTEP-1:0] dbresetout
,output [4:0] dbstate
,output [4:0] dbnext
,output [15:0] dbattemptcnt
,output [4:0] dbtxphase5
,output [31:0] dbcnt
);

localparam TIMEOUTSHORT=32'h4000000;
localparam TIMEOUTLONG=32'hc0000000;
//wire waitrxpcommaalignen;
wire RXUSERRDY=gticc.rxuserrdy;
wire TXUSERRDY=gticc.txuserrdy;
reg [DBYTE-1:0] txcharisk_r=0;//waitrxpcommaalignen ? 4'h1 : gticc.txcharisk;
reg [DWIDTH-1:0] txdata_r=0;//waitrxpcommaalignen ? 32'h01bc : gticc.txdata;
wire [DBYTE-1:0] txcharisk=txcharisk_r;
wire [DBYTE-1:0] RXCHARISK;
wire [DBYTE-1:0] RXCHARISCOMMA;
wire [DBYTE-1:0] RXDISPERR;
wire [DBYTE-1:0] RXNOTINTABLE;
wire RXELECIDLE;
wire [DWIDTH-1:0] RXDATA;
wire resetdone;
wire rxusrclk;
wire txusrclk;


//wire txrxreset1;
//wire txrxreset0=1'b0;
//wire txrxreset=txrxreset0|txrxreset1;
wire [63:0] txdata64;
wire [7:0] txcharisk8;
wire [63:0] rxdata64;
wire [7:0] rxdisperr8;
wire [7:0] rxcharisk8;
wire [7:0] rxchariscomma8;
wire [7:0] rxnotintable8;
wire [64-DWIDTH-1:0] rxdata_float_i;
wire [8-DBYTE-1:0] rxcharisk_float_i;
wire [8-DBYTE-1:0] rxchariscomma_float_i;
wire [8-DBYTE-1:0] rxdisperr_float_i;
wire [8-DBYTE-1:0] rxnotintable_float_i;
assign txcharisk8={{8-DBYTE-1{1'b0}},txcharisk};
assign txdata64={{64-DWIDTH{1'b0}},txdata_r} ;
assign {rxdata_float_i,RXDATA}=rxdata64;
assign {rxdisperr_float_i,RXDISPERR}=rxdisperr8;
assign {rxcharisk_float_i,RXCHARISK}=rxcharisk8;
assign {rxchariscomma_float_i,RXCHARISCOMMA}=rxchariscomma8;
assign {rxnotintable_float_i,RXNOTINTABLE}=rxnotintable8;
wire CPLLRESET;
wire GTTXRESET;//=txrxreset;
wire GTRXRESET;//=txrxreset;
wire RXRESETDONE;
wire TXRESETDONE;
wire CPLLFBCLKLOST;
wire CPLLLOCK;
wire CPLLREFCLKLOST;
wire RXBYTEISALIGNED;
wire RXBYTEREALIGN;
wire RXUSRCLK;
wire RXUSRCLK2;
wire TXUSRCLK;
wire TXUSRCLK2;
wire RXDLYSRESET;
//wire RXDLYEN=1'b0;
//wire RXPHALIGN=1'b0;
//wire RXPHALIGNEN=1'b0;
//wire RXPHDLYRESET=1'b0;
//wire RXLPMHFHOLD=1'b0;
//wire RXLPMLFHOLD=1'b0;

wire TXDLYSRESET;
wire TXPHDLYRESET;//=txrxreset;
wire TXPHINITDONE;
wire TXPHALIGNDONE;
wire TXDLYSRESETDONE;
wire [4:0] RXPHSLIPMONITOR;
wire [4:0] RXPHMONITOR;
wire RXPHALIGNDONE;
wire RXDLYSRESETDONE;
wire [2:0] TXOUTCLKSEL;

assign txusrclk=TXUSRCLK2;
assign rxusrclk=RXUSRCLK2;
//wire RXDISPERR;

wire TXOUTCLK;
wire RXOUTCLK;
reg cdrlocked=0;
wire mmcmlocked;
assign dblocked=cdrlocked;
reg [2:0] unlockcnt=0;
wire rxuserrdy=RXUSERRDY&cdrlocked&mmcmlocked;
wire RXCDRLOCK;
assign dbrxcdrlock=RXCDRLOCK;

`include "iccgt_1G_125M_50M.vh"

wire txoutclk2;
//BUFG txoutclk_bufg(.I(SIM ? txoutclk2 : TXOUTCLK),.O(TXUSRCLK));
assign TXUSRCLK2=TXUSRCLK;

wire mmcmreset;
wire mmcmclkfb;
wire rdyfortxrxreset;

generate
if (SIM) begin
	BUFG txoutclk_bufg(.I(txoutclk2),.O(TXUSRCLK));
	assign TXOUTCLKSEL=3'b011;//SIM ? 3'b011 : 3'b010;
	MMCME2_ADV#(.BANDWIDTH("OPTIMIZED"),.CLKFBOUT_MULT_F(5.0),.CLKOUT0_DIVIDE_F(10),.CLKIN1_PERIOD(8),.DIVCLK_DIVIDE(1)
	)mmcm_adv_inst
	(.CLKIN1(TXOUTCLK),.CLKIN2(1'b0),.CLKINSEL(1'b1)
	,.CLKOUT0(txoutclk2)
	,.CLKFBOUT(mmcmclkfb)
	,.CLKFBIN(mmcmclkfb)
	,.DADDR(7'h0),.DCLK(1'b0),.DEN(1'b0),.DI(16'h0),.DO(),.DRDY(),.DWE(1'b0)
	,.PSCLK(1'b0),.PSEN(1'b0),.PSINCDEC(1'b0),.PSDONE()
	,.LOCKED(mmcmlocked)
	,.RST(mmcmreset)
	,.PWRDWN(1'b0)
	);
end
else begin
	//BUFG txoutclk_bufg(.I(TXOUTCLK),.O(TXUSRCLK));
	BUFG txoutclk_bufg(.I(TXOUTCLK),.O(txoutclk));
	BUFG txusrclk_bufg(.I(phrefclk62_5),.O(TXUSRCLK));
	assign mmcmlocked=1'b1;
	assign TXOUTCLKSEL=3'b010;//SIM ? 3'b011 : 3'b010;
end
endgenerate

BUFG rxoutclk_bufg(.I(RXOUTCLK),.O(RXUSRCLK));
assign RXUSRCLK2=RXUSRCLK;
localparam LOCKMAX=3;
always @(posedge RXUSRCLK) begin
	if (gticc.rxreset) begin
		unlockcnt<=0;
		cdrlocked<=1'b0;
	end
	else begin
		if (~RXCDRLOCK) begin
			if (cdrlocked!=LOCKMAX) begin
				unlockcnt<=unlockcnt+1;
			end
			else begin
				cdrlocked<=1'b0;
			end
		end
		else begin
			unlockcnt<=0;
			cdrlocked<=1'b1;
		end
	end
end

reg txphaligndone_d=0;
reg txphaligndone_d2=0;
reg txphaligndone_d3=0;
reg txphaligndone_d4=0;
reg txphaligndone=0;
//wire txphaligndonerising=txphaligndone_d3&~txphaligndone_d4;
wire txphaligndonerising=TXPHALIGNDONE&~txphaligndone_d;
reg txphaligndone1st=0;
always @(posedge CPLLLOCKDETCLK) begin
	txphaligndone_d2<=txphaligndone_d;
	txphaligndone_d3<=txphaligndone_d2;
	txphaligndone_d4<=txphaligndone_d3;
end
always @(posedge TXUSRCLK) begin
	txphaligndone_d<=TXPHALIGNDONE;
	if (TXDLYSRESET|TXDLYSRESETDONE) begin
		txphaligndone1st<=1'b1;
		txphaligndone<=1'b0;
	end
	else begin
		if (txphaligndonerising) begin
			txphaligndone1st<=1'b0;
			if (~txphaligndone1st) begin
				txphaligndone<=1'b1;
			end
		end
	end
end
reg rxphaligndone_d=0;
reg rxphaligndone_d2=0;
reg rxphaligndone_d3=0;
reg rxphaligndone_d4=0;
reg rxphaligndone=0;
wire rxphaligndonerising=RXPHALIGNDONE&~rxphaligndone;
reg rxphaligndone1st=0;
always @(posedge CPLLLOCKDETCLK) begin
	rxphaligndone_d2<=rxphaligndone_d;
	rxphaligndone_d3<=rxphaligndone_d2;
	rxphaligndone_d4<=rxphaligndone_d3;
end
always @(posedge RXUSRCLK) begin
	rxphaligndone_d<=RXPHALIGNDONE;
	if (RXDLYSRESET|RXDLYSRESETDONE) begin
		rxphaligndone1st<=1'b1;
		rxphaligndone<=1'b0;
	end
	else begin
		if (rxphaligndonerising) begin
			rxphaligndone1st<=1'b0;
			if (~rxphaligndone1st) begin
				rxphaligndone<=1'b1;
			end
		end
	end
end

wire [DWIDTH-1:0] palign3data=16'h03bc;
wire [DBYTE-1:0] palign3isk=2'b01;
wire [DWIDTH-1:0] palign2data=16'h02bc;//{{(DWIDTH-8){1'b0}},8'hbc};
wire [DBYTE-1:0] palign2isk=2'b01;//{{(DBYTE-1){1'b0}},1'b1};
wire [DWIDTH-1:0] palign1data =16'h01bc;
wire [DBYTE-1:0] palign1isk=2'b01;
wire [DWIDTH-1:0] comma=16'h00bc;//{{(DWIDTH-8){1'b0}},8'hbc};
wire [DBYTE-1:0] commaisk=2'b01;//{{(DBYTE-1){1'b0}},1'b1};


wire clk=CPLLLOCKDETCLK;
wire align1req=RXDATA==palign1data & RXCHARISK==palign1isk;
wire align2req=RXDATA==palign2data & RXCHARISK==palign2isk;
wire align3req=RXDATA==palign3data & RXCHARISK==palign3isk;

reg reset_d=0;
wire [13:0] txphase14=txphase&14'h3fff;
wire [16:0] txphase17=((txphase14<<<2)+(txphase14));
wire [4:0] txphase5=txphase17>>>12;
wire [4:0] txphase5_x;
reg farendaligned=0;

reg txreq1=0;
reg txreq2=0;
reg txreq3=0;
reg txseldata=0;
wire sfplos_x,stb_txphase_x,mmcmlocked_x,bypasstxphcheck_x,txphaligndone_x,rxphaligndone_x,cdrlocked_x,farendaligned_x,align1req_x,align2req_x,align3req_x;
alatch #(.DWIDTH(5+9)) xdomaintostable(.clk(clk),.datain(
	{sfplos,stb_txphase,mmcmlocked,bypasstxphcheck,cdrlocked,farendaligned,align1req,align2req,align3req,txphase5}
),.dataout(
	{sfplos_x,stb_txphase_x,mmcmlocked_x,bypasstxphcheck_x,cdrlocked_x,farendaligned_x,align1req_x,align2req_x,align3req_x,txphase5_x}
));
areset txphaligndone_areset(.clk(clk),.areset(txphaligndone),.sreset(txphaligndone_x));
areset rxphaligndone_areset(.clk(clk),.areset(rxphaligndone),.sreset(rxphaligndone_x));
wire txreq1_x,txreq2_x,txreq3_x,txseldata_x;
alatch #(.DWIDTH(4)) xdomaintotx(.clk(txusrclk),.datain({txreq1,txreq2,txreq3,txseldata}),.dataout({txreq1_x,txreq2_x,txreq3_x,txseldata_x}));
//typedef
enum {IDLE//0
,STTXWAIT500NS//1
,STCPLLRESET//2
,STCPLLRESETING//3
,STMMCMRESETING//4
,STTXRXRESET//5
,STTXRXRESETING//6
,STTXRESETDONE//7
,STRXRESETDONE//8
,STTXRXRESETDONE//9

,STTXDLYSRESET//a
,STTXDLYSRESETD1//b
,STTXPHSTB1//c
,STTXPHSTB2//d
,STTXPHCHECK//e
,STRXWAIT500NS//f
,STRXCDRRDY//10
,STRXCDRSTABLE//11
,STRXDLYSRESET//12
,STRXDLYSRESETD1//13

,ALIGN1REQ//14
,ALIGN2REQ//15
,ALIGN2CHK//16
,ALIGN3REQ//17
,DATA//18
,RXERR//19
,TXTMO//1a
,RXTMO//1b
} state=IDLE,next=IDLE;
//fsmstate;
assign dbstate=state;
assign dbnext=next;
//fsmstate state=IDLE;
//fsmstate next=IDLE;

//alatch xdomaintorx(.clk(rxusrclk),.datain(),.dataout());
reg [31:0] cnt=0;
assign tmo=cnt==TIMEOUTSHORT;
assign tmolong=cnt==TIMEOUTLONG;
assign dbcnt=cnt;
areset resettxxdomain (.clk(txusrclk),.areset(reset),.sreset(gticc.txreset));
areset resetrxxdomain (.clk(rxusrclk),.areset(reset),.sreset(gticc.rxreset));
reg txphgood=0;
always @(posedge clk) begin
	txphgood<=bypasstxphcheck_x | (txphase5_x ==txphtarget) ;
end
always @(posedge clk) begin
	if (reset)begin
		state<=IDLE;
	end
	else begin
		state<=next;
		cnt<=(state==next) & (state !=IDLE) ? cnt+1 : 0;
	end
end
wire rxerr=|{RXDISPERR,RXNOTINTABLE};
always @(*) begin
	if (reset) begin
		next=IDLE;
	end
	else begin
		case (state)
			IDLE: next = STTXWAIT500NS ;//: IDLE;
			STTXWAIT500NS: next = tmo ? TXTMO : (cnt==200) ? STCPLLRESET : STTXWAIT500NS;
			STCPLLRESET: next = tmo ? TXTMO : cnt==2 ? STCPLLRESETING : STCPLLRESET;
			STCPLLRESETING: next =tmo ? TXTMO :  CPLLLOCK ? STMMCMRESETING : STCPLLRESETING;
			STMMCMRESETING: next =tmo ? TXTMO :  mmcmlocked_x ? STTXRXRESET : STMMCMRESETING;
			STTXRXRESET: next = tmo ? TXTMO : cnt==100 ? STTXRXRESETING : STTXRXRESET;
			STTXRXRESETING: next = tmolong ? TXTMO : TXRESETDONE ? STTXRESETDONE : RXRESETDONE ? STRXRESETDONE : STTXRXRESETING;
			STTXRESETDONE: next = tmo ? TXTMO : |{~CPLLLOCK, CPLLFBCLKLOST,CPLLREFCLKLOST} ? STTXWAIT500NS : RXRESETDONE ? STTXRXRESETDONE : STTXRESETDONE;
			STRXRESETDONE: next = tmo ? TXTMO : |{~CPLLLOCK, CPLLFBCLKLOST,CPLLREFCLKLOST} ? STTXWAIT500NS : TXRESETDONE ? STTXRXRESETDONE : STRXRESETDONE;
			STTXRXRESETDONE: next =  tmolong ? RXTMO :&{TXRESETDONE,RXRESETDONE,CPLLLOCK,~CPLLFBCLKLOST,~CPLLREFCLKLOST} ? STTXDLYSRESET : STTXWAIT500NS;

			STTXDLYSRESET: next =cnt==5 ? STTXDLYSRESETD1: STTXDLYSRESET;
			STTXDLYSRESETD1: next = tmo ? TXTMO : (txphaligndone_x | 1'b1 )? STTXPHSTB1 : STTXDLYSRESETD1;
			STTXPHSTB1: next = stb_txphase_x ? STTXPHSTB2 : STTXPHSTB1;
			STTXPHSTB2: next = stb_txphase_x ? STTXPHCHECK : STTXPHSTB2;
			STTXPHCHECK: next = txphgood ?  STRXWAIT500NS : STTXWAIT500NS;
			//STTXPHCHECK: next = txphgood ?  DATA : STTXWAIT500NS;
			STRXWAIT500NS: next = ~txphgood ? STTXWAIT500NS :  (cnt==200) ? STRXCDRRDY : STRXWAIT500NS;
			STRXCDRRDY: next= ~txphgood ? STTXWAIT500NS :  ~sfplos_x & cdrlocked_x ? STRXCDRSTABLE : STRXCDRRDY;
			STRXCDRSTABLE: next = ~txphgood ? STTXWAIT500NS :  ~cdrlocked_x|sfplos_x  ?  STRXCDRRDY :  cnt==32'hffffff ? STRXDLYSRESET :STRXCDRSTABLE  ;
			STRXDLYSRESET: next = tmo ? RXTMO : ~txphgood ? STTXWAIT500NS :  cnt==5 ? STRXDLYSRESETD1 : STRXDLYSRESET;
			STRXDLYSRESETD1: next= tmo ? RXTMO : rxphaligndone_x ? ALIGN1REQ : STRXDLYSRESETD1;


			ALIGN1REQ: next= tmo ? RXTMO : ~cdrlocked_x  ? STRXWAIT500NS  :  RXBYTEISALIGNED ? ALIGN2REQ : ALIGN1REQ;
			ALIGN2REQ:       next= tmo ? RXTMO : ~cdrlocked_x  ? STRXWAIT500NS  :~RXBYTEISALIGNED ? ALIGN1REQ : cnt==100 ? ALIGN2CHK : ALIGN2REQ;
			ALIGN2CHK: next= tmo ? RXTMO : ~cdrlocked_x  ? STRXWAIT500NS  : ~RXBYTEISALIGNED ? ALIGN1REQ : farendaligned_x  ? ALIGN3REQ : ALIGN2REQ;
			ALIGN3REQ: next= tmo ? RXTMO : ~cdrlocked_x  ? STRXWAIT500NS  : ~RXBYTEISALIGNED ? ALIGN1REQ : cnt==100 ? DATA : ALIGN3REQ;
			//ALIGN3REQ: next= tmo ? RXTMO : ~cdrlocked_x  ? STRXWAIT500NS  : ~RXBYTEISALIGNED ? ALIGN1REQ : cnt==100 ? STRXWAIT500NS : ALIGN3REQ;
			DATA: next = |{~CPLLLOCK,CPLLFBCLKLOST,CPLLREFCLKLOST,~mmcmlocked_x,~txphgood } ? STTXWAIT500NS :
//				~cdrlocked_x  ? STRXWAIT500NS :
//				rxerr ? RXERR :
//				~RXBYTEISALIGNED ? ALIGN1REQ :
//				align1req_x ? ALIGN2REQ :
//				align2req_x ? ALIGN3REQ :
				DATA ;
			RXERR: next= ALIGN1REQ;//STRXWAIT500NS ;
			TXTMO: next = STTXWAIT500NS;
			RXTMO: next = STTXWAIT500NS;
			default: next=STTXWAIT500NS;
		endcase
	end
end
reg [15:0] attempt_r=0;
reg [15:0] attemptcnt_r=0;
assign dbattemptcnt=attemptcnt_r;
assign dbtxphase5=txphase5;
reg cpllreset_r=0;
reg mmcmreset_r=0;
reg gttxrxreset_r=0;
reg rxdlysreset_r=0;
reg txdlysreset_r=0;
assign RXDLYSRESET=rxdlysreset_r;
assign TXDLYSRESET=txdlysreset_r;
reg done_r=0;
always @(posedge clk) begin
	if (reset) begin
	end
	else begin
		case (next)
			IDLE: begin
				txreq1<=1'b1;txreq2<=1'b0;txreq3<=1'b0;txseldata<=1'b0;
			end
			STTXWAIT500NS: begin
				cpllreset_r<=1'b1;
				mmcmreset_r<=1'b1;
				gttxrxreset_r<=1'b1;
				rxdlysreset_r<=1'b0;
				done_r<=0;
			end
			STCPLLRESET: begin
				cpllreset_r<=1'b1;
				if (cnt==1)
					attempt_r<=attempt_r+1;
				done_r<=0;
			end
			STCPLLRESETING: begin
				cpllreset_r<=1'b0;
			end
			STMMCMRESETING: begin
				mmcmreset_r<=1'b0;
				gttxrxreset_r<=1'b0;
			end
			STTXRXRESET: begin
				gttxrxreset_r<=1'b1;
				done_r<=1'b1;
			end
			STTXRXRESETING: begin
				gttxrxreset_r<=1'b0;
			end
			STTXRESETDONE,STRXRESETDONE: begin
				gttxrxreset_r<=1'b0;
			end
			STTXRXRESETDONE: begin
				gttxrxreset_r<=1'b0;
			end
			STTXPHSTB1: begin
			end
			STTXPHSTB2: begin
			end
			STTXPHCHECK: begin
			end
			STTXDLYSRESET: begin
				txdlysreset_r<= 1'b1;
				farendaligned<=1'b0;
			end
			STTXDLYSRESETD1: begin
				txdlysreset_r<=1'b0;
			end
			STRXWAIT500NS: begin
				txreq1<=1'b1;txreq2<=1'b0;txreq3<=1'b0;txseldata<=1'b0;
			end
			STRXDLYSRESET: begin
				rxdlysreset_r<=1'b1;
			end
			STRXDLYSRESETD1: begin
				rxdlysreset_r<=1'b0;
			end
			ALIGN1REQ: begin
				if (align2req_x) begin
					farendaligned<=1'b1;
				end
				txreq1<=1'b1;txreq2<=1'b0;txreq3<=1'b0;txseldata<=1'b0;
			end
			ALIGN2CHK,ALIGN2REQ: begin
				if (align2req_x) begin
					farendaligned<=1'b1;
				end
				txreq1<=1'b0;txreq2<=1'b1;txreq3<=1'b0;txseldata<=1'b0;
			end
			ALIGN3REQ: begin
				txreq1<=1'b0;txreq2<=1'b0;txreq3<=1'b1;txseldata<=1'b0;
			end
			DATA: begin
				attempt_r<=0;
				attemptcnt_r<=attempt_r;
				done_r<=1'b1;
				farendaligned<=1'b0;
				txreq1<=1'b0;txreq2<=1'b0;txreq3<=1'b0;txseldata<=1'b1;
			end
			RXERR: begin
				done_r<=1'b0;
			end
		endcase

	end
end
always @(posedge txusrclk) begin
	txdata_r <=  gticc.txdata;//txreq1_x ? palign1data : txreq2_x ? palign2data : txreq3_x ? palign3data : txseldata_x ? gticc.txdata : comma;
	txcharisk_r<=gticc.txcharisk;//txreq1_x ? palign1isk : txreq2_x ? palign2isk : txreq3_x ? palign3isk : txseldata_x ? gticc.txcharisk : commaisk;
//	txdata_r <=  txreq1_x ? palign1data : txreq2_x ? palign2data : txreq3_x ? palign3data : txseldata_x ? gticc.txdata : comma;
//	txcharisk_r<=txreq1_x ? palign1isk : txreq2_x ? palign2isk : txreq3_x ? palign3isk : txseldata_x ? gticc.txcharisk : commaisk;
end
assign CPLLRESET=cpllreset_r;
assign mmcmreset=mmcmreset_r;
assign GTTXRESET=gttxrxreset_r;
assign GTRXRESET=gttxrxreset_r;
assign resetdone=done_r;
assign gticc.txdataout=txdata_r;
assign gticc.txchariskout=txcharisk;
/*
wire txalign;
assign rdyfortxrxreset=&{CPLLLOCK,~CPLLFBCLKLOST,~CPLLREFCLKLOST};
wire txrxresetdone=&{TXRESETDONE,RXRESETDONE,rdyfortxrxreset,mmcmlocked};
wire rxalign=&{rxphaligndone,~RXNOTINTABLE,~RXDISPERR};
wire [NSTEP-1:0] done;
wire [NSTEP-1:0] resetout;
wire [NSTEP-1:0] resetoutmask=resetout&mask;
assign RXPCOMMAALIGNEN=1'b1;
//wire [NSTEP-1:0] donecriteria={1'b1,RXPHALIGNDONE,txrxresetdone,txrxresetdone,rdyfortxrxreset} | ~mask;
//assign {txalign,rxdlyreset,txrxreset1,txrxreset0,CPLLRESET}=resetout&mask;
//wire [NSTEP-1:0] donecriteria={rxalign,rxphaligndone,txphaligndone&RXCDRLOCK&~RXELECIDLE,txrxresetdone,txrxresetdone,rdyfortxrxreset} | ~mask;
wire [NSTEP-1:0] donecriteria={rxalign,rxphaligndone,txphaligndone&RXCDRLOCK&~RXELECIDLE,txrxresetdone,mmcmlocked,rdyfortxrxreset} | ~mask;
assign {waitrxpcommaalignen,RXDLYSRESET,TXDLYSRESET,txrxreset1,mmcmreset,CPLLRESET}=resetoutmask;
assign dbdonecriteria=donecriteria;
wire [NSTEP*16-1:0] readylength={{3{16'd10}},16'd100,16'd100,16'd100};//,16'd10,16'd10};
wire [NSTEP*16-1:0] resetlength={16'd10,16'd2,16'd2,{3{16'd10}}};//,16'd10,16'd10};
wire [NSTEP*16-1:0] resettodonecheck={NSTEP{16'd100}};//,16'd10,16'd10};
wire [NSTEP*32-1:0] resettimeout={NSTEP{32'h10000000}};
wire [NSTEP*32-1:0] donelength={NSTEP{16'd10}};
assign resetdone=&done & txrxresetdone;
reg [NSTEP-1:0] dbresetout_r=0;
reg lost_d=0;
reg lost=0;
always @(posedge CPLLLOCKDETCLK)  begin
	dbresetout_r<=resetoutmask;
	lost_d<=reset ? 1'b0 : lost;
	lost<= reset ? 1'b0 : CPLLREFCLKLOST;
end

assign dbresetout=dbresetout_r;
chainreset #(.NSTEP(NSTEP))
chainreset(.clk(CPLLLOCKDETCLK)
,.resetin(reset)
,.resetout(resetout)
,.donecriteria(donecriteria)
,.resetlength(resetlength)
,.readylength(readylength)
,.resettodonecheck(resettodonecheck)
,.resettimeout(resettimeout)
,.donelength(donelength)
,.done(done)
,.dbstate(dbstate)
,.dbnext(dbnext)
);
*/
//,TXPHINITDONE,TXPHALIGNDONE,TXDLYSRESETDONE,RXPHSLIPMONITOR,RXPHMONITOR,RXPHALIGNDONE,RXDLYSRESETDONE
assign reset=gticc.reset;//|(lost_d&~lost);
assign dbresetout={
	rxdlysreset_r
	,txdlysreset_r
	,gttxrxreset_r
	,mmcmreset_r
	,cpllreset_r
};
assign dbdone={RXBYTEISALIGNED
,RXDLYSRESETDONE,RXPHALIGNDONE,TXDLYSRESETDONE,TXPHALIGNDONE
,RXDISPERR,RXNOTINTABLE
,~RXELECIDLE,RXCDRLOCK,~CPLLFBCLKLOST,~CPLLREFCLKLOST
,TXRESETDONE,RXRESETDONE,mmcmlocked,CPLLLOCK};
assign gticc.resetdone=resetdone;
assign gticc.rxbyteisaligned=RXBYTEISALIGNED;
assign gticc.rxbyterealign=RXBYTEREALIGN;
assign gticc.rxcharisk=RXCHARISK;
assign gticc.rxdata=RXDATA;
assign gticc.rxdisperr=RXDISPERR;
assign gticc.rxnotintable=RXNOTINTABLE;
assign gticc.rxusrclk=rxusrclk;
assign gticc.txusrclk=txusrclk;
endmodule
