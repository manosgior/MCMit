module gticc_gt #
(parameter DWIDTH=16
,parameter GT_SIM_GTRESET_SPEEDUP = "TRUE"
,parameter RX_DFE_KL_CFG2_IN = 32'h301148AC
,parameter SIM_CPLLREFCLK_SEL = 3'b001
,parameter PMA_RSV_IN = 32'h00018480
,parameter PCS_RSVD_ATTR_IN = 48'h000000000000
,localparam DBYTE=DWIDTH/8
,localparam NSTEP=6
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
,igticc.gt gticc
,output RXOUTCLKFABRIC
,output TXOUTCLKFABRIC
,output RXOUTCLKPCS
,output TXOUTCLKPCS
,output dblocked
,output dbrxcdrlock
,output [NSTEP-1:0] dbdonecriteria
,input [NSTEP-1:0] mask
,output  [24:0] dbdone
,output [NSTEP-1:0] dbresetout
,output [2:0] dbstate
,output [2:0] dbnext
);

wire waitrxpcommaalignen;
wire RXUSERRDY=gticc.rxuserrdy;
wire TXUSERRDY=gticc.txuserrdy;
wire [DBYTE-1:0] TXCHARISK=waitrxpcommaalignen ? 4'h1 : gticc.txcharisk;
wire [DWIDTH-1:0] TXDATA=waitrxpcommaalignen ? 32'h01bc : gticc.txdata;
wire reset;
wire [DBYTE-1:0] RXCHARISK;
wire [DBYTE-1:0] RXCHARISCOMMA;
wire [DBYTE-1:0] RXDISPERR;
wire [DBYTE-1:0] RXNOTINTABLE;
wire RXELECIDLE;
wire [DWIDTH-1:0] RXDATA;
wire resetdone;
wire rxusrclk;
wire txusrclk;


wire txrxreset1;
wire txrxreset0=1'b0;
wire txrxreset=txrxreset0|txrxreset1;
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
assign txcharisk8=~resetdone ? 8'h1 : {{8-DBYTE-1{1'b0}},TXCHARISK};
assign txdata64=~resetdone ? 64'h000000bc : {{64-DWIDTH{1'b0}},TXDATA} ;
assign {rxdata_float_i,RXDATA}=rxdata64;
assign {rxdisperr_float_i,RXDISPERR}=rxdisperr8;
assign {rxcharisk_float_i,RXCHARISK}=rxcharisk8;
assign {rxchariscomma_float_i,RXCHARISCOMMA}=rxchariscomma8;
assign {rxnotintable_float_i,RXNOTINTABLE}=rxnotintable8;
wire RXCDRRESET=1'b0;
wire CPLLRESET;
wire GTTXRESET=txrxreset;
wire GTRXRESET=txrxreset;
wire RXPMARESET=txrxreset;
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
wire RXDLYEN=1'b0;
wire RXDLYSRESET;
wire RXPHALIGN=1'b0;
wire RXPHALIGNEN=1'b0;
wire RXPHDLYRESET=1'b0;
wire RXLPMHFHOLD=1'b0;
wire RXLPMLFHOLD=1'b0;

wire TXDLYEN=1'b0;
wire TXDLYSRESET;
wire TXPHALIGN=1'b0;
wire TXPHALIGNEN=1'b0;
wire TXPHDLYRESET=txrxreset;
wire TXPHINIT=1'b0;
wire TXPHINITDONE;
wire TXPHALIGNDONE;
wire TXDLYSRESETDONE;
wire RXPHSLIPMONITOR;
wire RXPHMONITOR;
wire RXPHALIGNDONE;
wire RXDLYSRESETDONE;
assign txusrclk=TXUSRCLK2;
assign rxusrclk=RXUSRCLK2;
//wire RXDISPERR;

wire TXOUTCLK;
wire RXOUTCLK;
reg locked=0;
wire mmcmlocked;
assign dblocked=locked;
reg [2:0] unlockcnt=0;
wire rxuserrdy=RXUSERRDY&locked&mmcmlocked;
wire RXCDRLOCK;
assign dbrxcdrlock=RXCDRLOCK;

`include "iccgt.vh"

wire txoutclk2;
//BUFG txoutclk_bufg(.I(TXOUTCLK),.O(TXUSRCLK));
BUFG txoutclk_bufg(.I(txoutclk2),.O(TXUSRCLK));
//BUFG txoutclk_bufg(.I(RXOUTCLK),.O(TXUSRCLK));
assign TXUSRCLK2=TXUSRCLK;

wire mmcmclkfb;
wire mmcmreset;
wire rdyfortxrxreset;
MMCME2_ADV#(.BANDWIDTH("OPTIMIZED"),.CLKFBOUT_MULT_F(5.0),.CLKOUT0_DIVIDE_F(10),.CLKIN1_PERIOD(8),.DIVCLK_DIVIDE(1)
)mmcm_adv_inst
(.CLKIN1(TXOUTCLK),.CLKIN2(1'b0),.CLKINSEL(1'b1)
,.CLKOUT0(txoutclk2)
,.CLKFBOUT(mmcmclkfb)
,.CLKFBIN(mmcmclkfb)
,.DADDR(7'h0),.DCLK(1'b0),.DEN(1'b0),.DI(16'h0),.DO(),.DRDY(),.DWE(1'b0)
,.PSCLK(1'b0),.PSEN(1'b0),.PSINCDEC(1'b0),.PSDONE()
,.LOCKED(mmcmlocked)
,.RST(mmcmreset|~rdyfortxrxreset)
,.PWRDWN(1'b0)
);


BUFG rxoutclk_bufg(.I(RXOUTCLK),.O(RXUSRCLK));
assign RXUSRCLK2=RXUSRCLK;
localparam LOCKMAX=3;
always @(posedge RXUSRCLK or posedge reset) begin
	if (reset) begin
		unlockcnt<=0;
		locked<=1'b0;
	end
	else begin
		if (~RXCDRLOCK) begin
			if (locked!=LOCKMAX) begin
				unlockcnt<=unlockcnt+1;
			end
			else begin
				locked<=1'b0;
			end
		end
		else begin
			unlockcnt<=0;
			locked<=1'b1;
		end
	end
end

reg txphaligndone_d=0;
reg txphaligndone=0;
wire txphaligndonerising=TXPHALIGNDONE&~txphaligndone_d;
reg txphaligndone1st=0;
always @(posedge CPLLLOCKDETCLK) begin
	txphaligndone_d<=TXPHALIGNDONE;
	if (txrxreset|TXDLYSRESET|TXDLYSRESETDONE) begin
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
reg rxphaligndone=0;
wire rxphaligndonerising=RXPHALIGNDONE&~rxphaligndone_d;
reg rxphaligndone1st=0;
always @(posedge CPLLLOCKDETCLK) begin
	rxphaligndone_d<=RXPHALIGNDONE;
	if (txrxreset|RXDLYSRESET|RXDLYSRESETDONE) begin
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
assign reset=gticc.reset;//|(lost_d&~lost);
assign dbdone={done,mmcmlocked,~RXELECIDLE,RXCDRLOCK
,txphaligndone,rxphaligndone,TXPHINITDONE,TXPHALIGNDONE
,TXDLYSRESETDONE,RXPHSLIPMONITOR,RXPHMONITOR,RXPHALIGNDONE
,RXDLYSRESETDONE,CPLLLOCK,~CPLLFBCLKLOST,~CPLLREFCLKLOST
,TXRESETDONE,RXRESETDONE,rdyfortxrxreset,txrxresetdone
};

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

