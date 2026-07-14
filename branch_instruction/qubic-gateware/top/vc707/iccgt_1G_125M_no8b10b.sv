module iccgt_1G_125M_no8b10b#
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
,igticc.gt gticc
,input bypasstxphcheck
//,output RXOUTCLKFABRIC
//,output TXOUTCLKFABRIC
//,output RXOUTCLKPCS
//,output TXOUTCLKPCS
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

localparam TIMEOUTSHORT= SIM ? 32'h80 : 32'h40000000;
localparam TIMEOUTLONG= SIM ? 32'h160 : 32'hc0000000;
//wire waitrxpcommaalignen;
wire RXUSERRDY=gticc.rxuserrdy;
wire TXUSERRDY=gticc.txuserrdy;
reg [DWIDTH-1:0] txdata_r=0;//waitrxpcommaalignen ? 32'h01bc : gticc.txdata;
wire reset;
assign reset=gticc.reset;//|(lost_d&~lost);
wire resetdone;
wire rxusrclk;
wire txusrclk;


wire [63:0] rxdata64;
wire [7:0] rxdisperr8;
wire [7:0] rxcharisk8;
wire [7:0] txchardispmode8;
wire [7:0] txchardispval8;
wire [63:0] txdata64;
wire [64-DWIDTH-1:0] rxdata_float_i;
wire [8-DBYTE-1:0] rxcharisk_float_i;
wire [8-DBYTE-1:0] rxdisperr_float_i;
wire [2:0] TXOUTCLKSEL;
`include "iccgt_1G_125M_no8b10b.vh"

wire [19:0] txdata20;//=20'h01cfa;
wire [19:0] rxdata20;

assign txdata64={48'b0,txdata20[17:10],txdata20[7:0]};
assign txchardispval8={6'b0,txdata20[18],txdata20[8]};
assign txchardispmode8={6'b0,txdata20[19],txdata20[9]};
assign rxdata20={rxdisperr8[1],rxcharisk8[1],rxdata64[15:8],rxdisperr8[0],rxcharisk8[0],rxdata64[7:0]};
//assign txdata20=20'haaaaa;

assign txusrclk=TXUSRCLK2;
assign rxusrclk=TXUSRCLK2;
reg rxaligned=0;
wire rxdone;
assign rxdone=&{rxaligned,RXRESETDONE};
reg txreset=1'b1;
ifcodec8b10b codec(.txclk(txusrclk),.rxclk(txusrclk),.rxreset(~rxdone),.txreset(txreset));
//wire RXDISPERR;

wire TXOUTCLK;
wire RXOUTCLK;
reg cdrlocked=0;
wire mmcmtxlocked;
wire mmcmrxlocked;
wire mmcmlocked=mmcmtxlocked;//&mmcmrxlocked;
assign dblocked=cdrlocked;
reg [2:0] unlockcnt=0;
wire rxuserrdy=RXUSERRDY&cdrlocked&mmcmlocked;
//assign dbrxcdrlock=RXCDRLOCK;


wire rxoutclk62_5;
wire rxoutclk125;
wire txoutclk62_5;
wire txoutclk125;
//BUFG txoutclk_bufg(.I(SIM ? txoutclk2 : TXOUTCLK),.O(TXUSRCLK));
//assign TXUSRCLK2=TXUSRCLK;

wire mmcmreset;
wire mmcmtxclkfb;
wire mmcmrxclkfb;
wire rdyfortxrxreset;

generate
if (SIM) begin
	BUFG txoutclk_bufg(.I(txoutclk62_5),.O(TXUSRCLK));
	BUFG txusrclk2_bufg(.I(txoutclk125),.O(TXUSRCLK2));
	assign TXOUTCLKSEL=3'b011;//SIM ? 3'b011 : 3'b010;
	MMCME2_ADV#(.BANDWIDTH("OPTIMIZED"),.CLKFBOUT_MULT_F(10.0),.CLKOUT0_DIVIDE_F(10),.CLKIN1_PERIOD(16),.DIVCLK_DIVIDE(1),.CLKOUT1_DIVIDE(5))mmcmtx
	(.CLKIN1(TXOUTCLK),.CLKIN2(1'b0),.CLKINSEL(1'b1)
	,.CLKOUT0(txoutclk62_5)
	,.CLKOUT1(txoutclk125)
	,.CLKFBOUT(mmcmtxclkfb)
	,.CLKFBIN(mmcmtxclkfb)
	,.DADDR(7'h0),.DCLK(1'b0),.DEN(1'b0),.DI(16'h0),.DO(),.DRDY(),.DWE(1'b0)
	,.PSCLK(1'b0),.PSEN(1'b0),.PSINCDEC(1'b0),.PSDONE()
	,.LOCKED(mmcmtxlocked)
	,.RST(mmcmreset)
	,.PWRDWN(1'b0)
	);

end
else begin
	BUFG txoutclk_bufg(.I(TXOUTCLK),.O(TXUSRCLK));
	assign mmcmlocked=1'b1;
	assign TXOUTCLKSEL=3'b010;//SIM ? 3'b011 : 3'b010;
end
endgenerate

wire [19:0] rxdata20_x;
wire rxfifoempty;
wire rxfifofull;

fifo#(.DW(20),.AW(2),.SIM(SIM),.SAMECLKDOMAIN(1))
fiforxd(.wclk(RXUSRCLK),.rclk(TXUSRCLK),.din(rxdata20),.wr_en(1'b1),.rd_en(1'b1),.dout(rxdata20_x),.empty(rxfifoempty),.full(rxfifofull)
,.rst(1'b0)
);

/*
assign rxdata20_x=rxdata20;
assign rxfifoempty=1'b0;
assign rxfifofull=1'b0;
*/

reg [4:0] shiftpos=0;
reg [39:0] rxdatasr40=0;
wire [39:0] rxdata40_shifted=rxdatasr40<<shiftpos;
reg [19:0] rxdata20_shifted=0;
always @(posedge TXUSRCLK) begin
	rxdatasr40<=~RXRESETDONE ? 0 : {(rxfifoempty ? 20'd0 : rxdata20_x),rxdatasr40[39:20]};
	//rxdatasr40<=~RXRESETDONE ? 0 : {(rxfifoempty ? 20'd0 : 20'd0),rxdatasr40[39:20]};
	{shiftpos,rxaligned}<=~RXRESETDONE ? 0 :
				(rxdatasr40[39:30]==10'h17c) ? {5'd10,1'b1} :  // only check pcomma not check mcomma : rxdatasr40[]==10'h283
				(rxdatasr40[38:29]==10'h17c) ? {5'd11,1'b1} :
				(rxdatasr40[37:28]==10'h17c) ? {5'd12,1'b1} :
				(rxdatasr40[36:27]==10'h17c) ? {5'd13,1'b1} :
				(rxdatasr40[35:26]==10'h17c) ? {5'd14,1'b1} :
				(rxdatasr40[34:25]==10'h17c) ? {5'd15,1'b1} :
				(rxdatasr40[33:24]==10'h17c) ? {5'd16,1'b1} :
				(rxdatasr40[32:23]==10'h17c) ? {5'd17,1'b1} :
				(rxdatasr40[31:22]==10'h17c) ? {5'd18,1'b1} :
				(rxdatasr40[30:21]==10'h17c) ? {5'd19,1'b1} :
				(rxdatasr40[29:20]==10'h17c) ? {5'd0,1'b1} :
				(rxdatasr40[28:19]==10'h17c) ? {5'd1,1'b1} :
				(rxdatasr40[27:18]==10'h17c) ? {5'd2,1'b1} :
				(rxdatasr40[26:17]==10'h17c) ? {5'd3,1'b1} :
				(rxdatasr40[25:16]==10'h17c) ? {5'd4,1'b1} :
				(rxdatasr40[24:15]==10'h17c) ? {5'd5,1'b1} :
				(rxdatasr40[23:14]==10'h17c) ? {5'd6,1'b1} :
				(rxdatasr40[22:13]==10'h17c) ? {5'd7,1'b1} :
				(rxdatasr40[21:12]==10'h17c) ? {5'd8,1'b1} :
				(rxdatasr40[20:11]==10'h17c) ? {5'd9,1'b1} :
				{shiftpos,rxaligned};
	rxdata20_shifted<=rxaligned ? rxdata40_shifted[39:20] : 0;
	//rxdata20_shifted<=0;//rxaligned ? rxdata40_shifted[39:20] : 0;
end
reg rx2sw=0;
reg tx2sw=0;
reg [9:0] rxdata10=0;
reg [19:0] txdata20_r=0;
reg [19:0] txdata20sr=0;
reg [19:0] txdata20sr_d=0;
reg [15:0] rxdata16sr=0;
reg [15:0] rxdata16sr_d=0;
reg [15:0] rxdata16=0;
reg [31:0] txcnt=0;
reg [15:0] txdata16=0;
reg [1:0] txdata16isk=0;
reg [1:0] rxdataisksr=0;
reg [1:0] rxdataisk=0;
reg [1:0] rxdataisksr_d=0;
always @(posedge TXUSRCLK2) begin
//	txcnt<=txcnt+1;
	rx2sw<=~rx2sw;
	rxdata10<=~TXUSRCLK ? rxdata20_shifted[19:10] : rxdata20_shifted[9:0];
	tx2sw<=~tx2sw;
	txdata20sr<={codec.tx10b,txdata20sr[19:10]};
	txdata20sr_d<=txdata20sr;
	rxdata16sr<={rxdata16sr[7:0],codec.rx8b};
	rxdata16sr_d<=rxdata16sr;
	rxdataisksr<={rxdataisksr[0],codec.rx8bisk};
	rxdataisksr_d<=rxdataisksr;
end
always @(posedge TXUSRCLK) begin
	txdata16<=gticc.txdata;//txcnt[5:1]==0 ? 16'h5cbc : txcnt[16:1];//{8'hf0,txcnt[8:1]};
	txdata16isk<=gticc.txcharisk;//txcnt[5:1]==0 ? 2'b11 : 0;
	txdata20_r<=txdata20sr_d;
	rxdata16<=rxdata16sr;
	rxdataisk<=rxdataisksr;
end
assign txdata20=txdata20_r;

assign codec.tx8b= TXUSRCLK ? txdata16[7:0] : txdata16[15:8];//8'hbc : 8'h0;
assign codec.tx8bisk= TXUSRCLK ? txdata16isk[0] : txdata16isk[1];//1'b1 : 1'b0;
assign codec.rx10b=rxdata10;


BUFG rxoutclk_bufg(.I(RXOUTCLK),.O(RXUSRCLK));
assign RXUSRCLK2=RXUSRCLK;

wire detclk=CPLLLOCKDETCLK;
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
,DATA//A
,STEPS
} state=IDLE,next=IDLE;

localparam WSTEPS=$clog2(STEPS)+1;
reg [31:0] cnt=0;
assign tmo=cnt==TIMEOUTSHORT;
assign tmolong=cnt==TIMEOUTLONG;
reg txreq1=0;
reg txreq2=0;
reg txreq3=0;
reg txseldata=0;
reg cpllreset_r=0;
reg mmcmreset_r=0;
reg gttxrxreset_r=0;
reg rxdlysreset_r=0;
reg txdlysreset_r=0;
assign RXDLYSRESET=rxdlysreset_r;
assign TXDLYSRESET=txdlysreset_r;
reg done_r=0;
reg [15:0] attempt_r=0;
reg [15:0] attemptcnt_r=0;
wire sreset;
areset resetxdomain (.clk(detclk),.areset(reset),.sreset(sreset));
always @(posedge detclk) begin
	if (sreset)begin
		state<=IDLE;
	end
	else begin
		state<=next;
		cnt<=(state==next) & (state !=IDLE) ? cnt+1 : 0;
	end
end
always @(*) begin
	if (sreset) begin
		next=IDLE;
	end
	else begin
		case (state)
			IDLE: next = STTXWAIT500NS ;//: IDLE;
			STTXWAIT500NS: next = (cnt==200) ? STCPLLRESET : STTXWAIT500NS;
			STCPLLRESET: next = cnt==2 ? STCPLLRESETING : STCPLLRESET;
			STCPLLRESETING: next =CPLLLOCK ? STMMCMRESETING : STCPLLRESETING;
			STMMCMRESETING: next =mmcmlocked ? STTXRXRESET : STMMCMRESETING;
			STTXRXRESET: next = cnt==100 ? STTXRXRESETING : STTXRXRESET;
			STTXRXRESETING: next = TXRESETDONE ? STTXRESETDONE : RXRESETDONE ? STRXRESETDONE : STTXRXRESETING;
			STTXRESETDONE: next = |{~CPLLLOCK, CPLLFBCLKLOST,CPLLREFCLKLOST} ? STTXWAIT500NS : RXRESETDONE ? STTXRXRESETDONE : STTXRESETDONE;
			STRXRESETDONE: next = |{~CPLLLOCK, CPLLFBCLKLOST,CPLLREFCLKLOST} ? STTXWAIT500NS : TXRESETDONE ? STTXRXRESETDONE : STRXRESETDONE;
			STTXRXRESETDONE: next =  &{TXRESETDONE,RXRESETDONE,CPLLLOCK,~CPLLFBCLKLOST,~CPLLREFCLKLOST} ? DATA : STTXWAIT500NS;
			DATA: next = DATA;
		endcase
	end
end

always @(posedge detclk) begin
	if (sreset) begin
	end
	else begin
		case (next)
			IDLE: begin
				txreq1<=1'b1;txreq2<=1'b0;txreq3<=1'b0;txseldata<=1'b0;
				txreset<=1'b1;
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
			end
			STTXRXRESETING: begin
				gttxrxreset_r<=1'b0;
			end
			STTXRESETDONE,STRXRESETDONE: begin
				gttxrxreset_r<=1'b0;
			end
			STTXRXRESETDONE: begin
				gttxrxreset_r<=1'b0;
				done_r<=1'b1;
			end
			DATA: begin
				txreset<=1'b0;
			end
		endcase
	end
end

assign CPLLRESET=cpllreset_r;
assign mmcmreset=mmcmreset_r;
assign GTTXRESET=gttxrxreset_r;
assign GTRXRESET=gttxrxreset_r;
assign resetdone=done_r;
assign gticc.txdataout=txdata16;
assign gticc.txchariskout=txdata16isk;

assign gticc.rxusrclk=rxusrclk;
assign gticc.txusrclk=txusrclk;

endmodule
