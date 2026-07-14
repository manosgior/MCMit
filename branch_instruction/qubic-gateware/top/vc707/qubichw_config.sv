module qubichw_config #(parameter DEBUG="false",parameter BAUD=9600,parameter SIM=0)
(hw hw
,regmap.cfg lbreg
,dsp.cfg dsp
);
`define JESD

enum {SYSCLKMMCM_RESET
,IDELAYCTRL_RESET
,PHYRESET
,PHYRESETWAIT5MS
,SGMIIETH_RESET
,UARTRESET
,UARTLBRESET
,I2CRESET
,MDIORESET
,I2CINITRESET_0
,LOADMACIP
,ETHRESET_W
,I2CINITRESET_1
,AXIRESET
,JESDRESET0
,JESDRESET1
,AXIINITRESET
,I2CINITRESET_2
,I2CINITRESET_3
,RESET_SFP
,RESETDONE
,RESET_SMASFP
,NSTEP
} chainnode;

//localparam QPLLRESET_113=16;
//localparam QPLLRESET_114=20;
//localparam HELPPLL=17;
//localparam NSTEP=20;
wire  clk100;
wire  clk125;
wire  clk200;
wire  clk250;
// sysclkmmcm cross wire
wire [31:0] clk100cnt;
wire [31:0] clk125cnt;
wire [31:0] clk200cnt;
wire [31:0] clk250cnt;
wire ethclk;
reg [31:0] ethclkcnt=0;
always @(posedge ethclk) begin
	ethclkcnt<=ethclkcnt+1;
end
wire sysclkmmcm_locked;
wire sysclkmmcm_reset;
sysclkmmcm sysclkmmcm(.clk100(clk100),.clk125(clk125),.clk200(clk200),.clk250(clk250),.clk100cnt(clk100cnt),.clk125cnt(clk125cnt),.clk200cnt(clk200cnt),.clk250cnt(clk250cnt),.sysclk(hw.vc707.sysclk),.mmcm_locked(sysclkmmcm_locked),.mmcm_reset(sysclkmmcm_reset));

localparam POWERONRESET=SIM ? 10 : 100000000;
reg poweronreset_r=0;
reg poweronreset_d=0;
wire poweronreset;
reg [31:0] poweronresetcnt=0;
reg reseted125=0;
always @(posedge hw.vc707.sysclk) begin
	poweronresetcnt<=poweronresetcnt+(poweronreset_r ? 1'b0 : 1'b1);
	if (poweronresetcnt==POWERONRESET & ~poweronreset_r)
		poweronreset_r<=1'b1;
	poweronreset_d<=poweronreset_r;
	if (poweronreset)
		reseted125=1'b1;
end
assign poweronreset = poweronreset_r& ~poweronreset_d;

ilocalbus #(.LBCWIDTH(8),.LBAWIDTH(24),.LBDWIDTH(32),.WRITECMD(1),.READCMD(8'h0))
uartlb();
ilocalbus #(.LBCWIDTH(8),.LBAWIDTH(24),.LBDWIDTH(32),.WRITECMD(0),.READCMD(8'h10))
udplb();
assign lbreg.lb.clk=udplb.clk;
assign lbreg.lb.wcmd=udplb.wcmd;
assign lbreg.lb.wvalid=udplb.wvalid;
assign udplb.rctrl=lbreg.lb.rctrl;
assign udplb.rdata=lbreg.lb.rdata;
assign udplb.raddr=lbreg.lb.raddr;
assign udplb.rready=lbreg.lb.rready;
assign lbreg.lb.readcmd=udplb.READCMD;
assign lbreg.lb.writecmd=udplb.WRITECMD;
assign udplb.wen=lbreg.lb.wen;
iuart_regmap#(.LBCWIDTH(8),.LBAWIDTH(24),.LBDWIDTH(32))
uartreg();
assign uartreg.lb.clk=uartlb.clk;
assign uartreg.lb.wcmd=uartlb.wcmd;
assign uartreg.lb.wvalid=uartlb.wvalid;
assign uartlb.rcmd=uartreg.lb.rcmd;
assign uartlb.rready=uartreg.lb.rready;
assign uartreg.lb.readcmd=uartlb.READCMD;
assign uartreg.lb.writecmd=uartlb.WRITECMD;


wire uarttx;
wire uartrx=hw.vc707.usb2uart.rx;
wire keeplbdataout;
wire keepadc;
assign hw.vc707.usb2uart.tx=uartreg.uartmode==0 ? hw.vc707.usb2uart.rx : uartreg.uartmode==1 ? uarttx : 1'b1; //serial port loopback test
assign {hw.vc707.gpio_led_7,hw.vc707.gpio_led_6,hw.vc707.gpio_led_5,hw.vc707.gpio_led_4,hw.vc707.gpio_led_3,hw.vc707.gpio_led_2,hw.vc707.gpio_led_1}={keepadc,keeplbdataout,clk200cnt[26:21]};
wire [7:0] rxdata;
wire [7:0] txdata;
wire txstart;
wire rxvalid;
wire [1:0] txstate;
wire [1:0] txnext;
wire txready;
wire txbaudclk;
wire txstop;
wire startfromtx;
wire [5:0] txtxcnt;
wire rxbaudclk;
wire [1:0] rxstate;
wire [1:0] rxnext;
wire [5:0] rxrxcnt;
wire [15:0] rxbaudcnt;
wire [15:0] txbaudcnt;
wire txline_r;
wire rxline_r;
wire uartclk=ethclk;//clk125;//cfg;
wire uartreset;
wire uartreset_x;
wire idelayctrl_reset;
wire dbresetcmd;
//flag_xdomain uartresetxdomain(.clk2(uartclk),.clk1(hw.vc707.sysclk),.flagin_clk1(uartreset),.flagout_clk2(uartreset_x));
areset uartareset(.clk(uartclk),.areset(uartreset),.sreset(uartreset_x));
uart #(.DWIDTH(8),.NSTOP(1),.UARTCLK(125000000),.BAUD(BAUD))
uart (.clk(uartclk),.TX(uarttx),.RX(uartrx),.rst(uartreset_x),.txdata(txdata),.txstart(txstart),.rxdata(rxdata),.rxvalid(rxvalid),.txready(txready));
wire uartlbreset;
wire uartlbreset_x;
//flag_xdomain uartlbresetxdomain(.clk2(uartclk),.clk1(hw.vc707.sysclk),.flagin_clk1(uartlbreset),.flagout_clk2(uartlbreset_x));
areset uartlbareset(.clk(uartclk),.areset(uartlbreset),.sreset(uartlbreset_x));
uartlb #(.UARTDWIDTH(8),.LBWIDTH(64))
uartlb64 (.clk(uartclk),.reset(uartlbreset_x),.fromuartdata(rxdata),.fromuartvalid(rxvalid),.touartdata(txdata),.touartready(txready),.touartstart(txstart),.lbrcmd(uartlb.rcmd),.lbrready(uartlb.rready),.lbwcmd(uartlb.wcmd),.lbwvalid(uartlb.wvalid)
,.dbresetcmd(dbresetcmd)
);

wire idelayctrl_rdy_w;
reg idelayctrl_rdy=0;
IDELAYCTRL idelayctrl(.RST(idelayctrl_reset),.RDY(idelayctrl_rdy_w),.REFCLK(hw.vc707.sysclk));
always @(posedge hw.vc707.sysclk) begin
	idelayctrl_rdy<=idelayctrl_rdy_w;
end

assign uartlb.clk=uartclk;
assign lbreg.fmcprsnt={hw.fmc2.prsnt,hw.fmc1.prsnt};
assign lbreg.fmcpgm2c={hw.fmc2.pg_m2c,hw.fmc1.pg_m2c};
assign {hw.fmc2.dac_txen_vadj,hw.fmc1.dac_txen_vadj}=lbreg.fmcdacen;
reg [31:0] i2cdatarx_r=0;
wire [31:0] i2cdatarx_w;
reg [31:0] i2cdatarx_wd;
wire sdatx;
wire sdarx;
wire scl=hw.vc707.iic.scl;
wire sdaasrx;
wire i2creset;
wire i2cresetdone;
wire  dbscl=hw.vc707.iic.scl;
wire [3:0] nack;
wire stopbit;
wire [31:0] i2cdatatx;
wire [31:0] i2cdatarx;
wire i2crxvalid;
wire i2cstart;
wire [3:0] i2cinitdone;
wire [36:0] i2ccmd;
wire [36:0] fmci2ccmd;
reg [36:0] i2ccmd_r=0;
reg [36:0] i2ccmd_rd=0;
reg [32:0] i2cresult=0;
wire [3:0] i2cinitreset;
/*reg [3:0] i2cinitreset_d=0;
reg i2creset_d=0;
always @(posedge hw.vc707.sysclk) begin
	i2cinitreset_d<=i2cinitreset;
	i2creset_d<=i2creset;
end
*/
wire si2cinitreset;
wire i2cbusy;
reg i2cbusy_d=0;
wire uarti2c_w;
reg uarti2c_r=0;
reg [31:0] uartreg__clk4ratio=0;
reg [31:0] lbreg__clk4ratio=0;
areset uari2cxdomain(.clk(ethclk),.areset(uartreg.uarti2c),.sreset_val(uarti2c_w));
reg mux_reset_b=0;
assign hw.vc707.iic.mux_reset_b=mux_reset_b;
always @(posedge ethclk) begin
	uarti2c_r<=uarti2c_w;
	uartreg__clk4ratio<=uartreg.clk4ratio;
	lbreg__clk4ratio<=lbreg.clk4ratio;
	mux_reset_b<=lbreg.i2cmux_reset_b | uartreg.i2cmux_reset_b | ~&i2cinitdone;
end

wire [32-1:0] datatx_run;
wire [3:0] nack_run;
wire stopbit_run;
wire [31:0] clk4ratio=SIM ? 2 : uarti2c_r ? uartreg__clk4ratio : lbreg__clk4ratio;
IOBUF sdaiobuf (.IO(hw.vc707.iic.sda),.I(sdatx),.O(sdarx),.T(sdaasrx));
wire i2cmasterreset_x;
areset i2cmasterareset(.clk(ethclk),.areset(i2creset),.sreset(i2cmasterreset_x));
i2cmaster #(.MAXNACK(4))
i2cmaster (.clk(ethclk),.sdatx(sdatx),.sdarx(sdarx),.sdaasrx(sdaasrx),.scl(hw.vc707.iic.scl),.clk4ratio(clk4ratio),.nack(nack),.stopbit(stopbit),.datatx(i2cdatatx),.start(i2cstart),.datarx(i2cdatarx),.rxvalid(i2crxvalid),.resetdone(i2cresetdone),.rst(i2cmasterreset_x),.busy(i2cbusy),.datatx_run(datatx_run),.nack_run(nack_run),.stopbit_run(stopbit_run)
);
wire [37:0] cfgi2ccmd;
wire i2cinitstart;
wire fmci2cbusy;
wire [31:0] fmci2cdatatx;
wire [3:0] fmci2cnack;
wire fmci2creset;
wire fmci2crunning;
wire [27:0] fmci2cdevcmd_2;
wire [27:0] fmci2cdevcmd_3;
wire fmci2cstb_devcmd_2;
wire fmci2cstb_devcmd_3;
wire [27:0] fmci2cdevcmd_1;
wire fmci2cstb_devcmd_1;
wire [27:0] fmci2cdevcmd=~i2cinitdone[1] ? fmci2cdevcmd_1 : (~i2cinitdone[2]) ? fmci2cdevcmd_2 : (~i2cinitdone[3]) ? fmci2cdevcmd_3 :0;
wire fmci2cstb_devcmd=~i2cinitdone[1] ? fmci2cstb_devcmd_1 : (~i2cinitdone[2]) ? fmci2cstb_devcmd_2 : (~i2cinitdone[3]) ? fmci2cstb_devcmd_3:0;
wire fmci2cstart;
wire fmci2cstopbit;
reg [37:0] i2c_x=0;
wire [37:0] i2c_s;
wire [37:0] i2c_v;
wire uarti2c_s=uartreg.stb_i2cstart;
wire [36:0] uarti2c_v={uartreg.i2cstart[4],uartreg.i2cstart[3:0],uartreg.i2cdatatx};
//areset #(.WIDTH(38)) i2cxdomain(.clk(ethclk),.areset(i2c_x),.sreset(i2c_s),.sreset_val(i2c_v));
assign {i2cstart,stopbit,nack,i2cdatatx}={i2c_x[37],i2c_x[36:0]};
always @(posedge ethclk) begin
i2c_x<=~i2cinitdone[0] ? {i2cinitstart,i2ccmd} :
	(~i2cinitdone[1])|(~i2cinitdone[2])|(~i2cinitdone[3]) ? {fmci2cstart,fmci2ccmd} :
	uartreg.uarti2c ? {uarti2c_s,uarti2c_v} : //{uartreg.stb_i2cstart,uartreg.i2cstart[4],uartreg.i2cstart[3:0],uartreg.i2cdatatx} :
	lbreg.lbi2c ?    {lbreg.stb_i2cstart,lbreg.i2cstart[4],lbreg.i2cstart[3:0],lbreg.i2cdatatx} :
	cfgi2ccmd;
end
//data_xdomain #(.DWIDTH(32+4+1))
//uarti2cxdomain(.clkin(clk125),.clkout(ethclk),.gatein(uartreg.stb_i2cstart),.datain({uartreg.i2cstart[4],uartreg.i2cstart[3:0],uartreg.i2cdatatx} ),.gateout(uarti2c_s),.dataout(uarti2c_v));
wire [3:0] dbi2cstate;
wire [3:0] dbi2cnext;
wire dbsi2cinitreset;
`include "i2cinit.vh"

wire smallchange;
wire [2:0] hs_div;
wire [6:0] n1;
wire [37:0] rfreq;
reg signed [37:0] rfreqfdbk=0;
reg stb_rfreqfdbk=0;
wire si570updatebusy;
wire [36:0] si570i2ccmd;
wire si570i2cstart;
wire [31:0] si57078=lbreg.si57078;
wire [31:0] si5709abc=lbreg.si5709abc;
wire [37:0] dbrfreq_w;
wire [37:0] dbsmallmax;
wire [37:0] dbsmallmin;
reg [5:0] newnow=0;
wire [5:0] dbnewnow;
reg [2:0] hs_div_now=0;
reg [6:0] n1_now=0;
reg [37:0] rfreq_now=0;
wire dbupdateext;
reg [7:0] i2cswlocation=0;
wire [5:0] debnewnow;
reg si570lbstart=0;
reg si570start_p=0;
reg lb570processed=1'b0;
reg si570updatebusy_d=1'b0;
always @(posedge ethclk) begin
	si570updatebusy_d<=si570updatebusy;
	if (lbreg.stb_si5709abc) begin
		si570lbstart<= 1'b1;
		lb570processed<=1'b0;
	end
	else begin
		if(~si570updatebusy & si570updatebusy_d ) begin
			lb570processed<=1'b1;
			if (lb570processed) begin
				si570lbstart<= 1'b0;
			end
		end
	end
//	si570lbstart<=si570start_p;
end
assign {smallchange,hs_div,n1,rfreq} = si570lbstart ? {si57078[16:0],si5709abc} : {1'b1,hs_div_now,n1_now,rfreqfdbk};
wire [3:0] dbstatesi570vc707;
wire [3:0] dbnextsi570vc707;
si570vc707 si570vc707(.clk(ethclk),.hs_div(hs_div),.n1(n1),.rfreq(rfreq),.start(si570lbstart | stb_rfreqfdbk),.smallchange(smallchange),.busy(si570updatebusy)
,.i2ccmd(si570i2ccmd),.i2cstart(si570i2cstart),.i2cbusy(i2cbusy)
,.hs_div_now(hs_div_now),.n1_now(n1_now),.rfreq_now(rfreq_now),.newnow(newnow)
,.dbrfreq_w(dbrfreq_w),.dbsmallmax(dbsmallmax),.dbsmallmin(dbsmallmin),.dbnewnow(dbnewnow),.dbstate(dbstatesi570vc707),.dbnext(dbnextsi570vc707)
);
wire updatesi570=i2cswlocation==8'h1 & datatx_run[31:24] =={7'h5d,1'h0} & nack_run==3 & stopbit_run;
always @(posedge ethclk) begin
	if (updatesi570) begin
        case (i2cdatatx[23:16])
            8'h7: begin {hs_div_now,n1_now[6:2]}<=i2cdatatx[15:8]; newnow[0]<=1'b1; end
            8'h8: begin {n1_now[1:0],rfreq_now[37:32]}<=i2cdatatx[15:8]; newnow[1]<=1'b1; end
            8'h9: begin rfreq_now[31:24]<=i2cdatatx[15:8]; newnow[2]<=1'b1; end
            8'ha: begin rfreq_now[23:16]<=i2cdatatx[15:8]; newnow[3]<=1'b1; end
            8'hb: begin rfreq_now[15:8]<=i2cdatatx[15:8]; newnow[4]<=1'b1; end
            8'hc: begin rfreq_now[7:0]<=i2cdatatx[15:8]; newnow[5]<=1'b1; end
        endcase
    end
end

assign cfgi2ccmd={si570i2cstart,si570i2ccmd};

wire i2cinitreset_x;
areset i2cinitareset(.clk(ethclk),.areset(i2cinitreset[0]),.sreset(i2cinitreset_x));
seqinit #(.INITWIDTH(32+4+1),.INITLENGTH(I2CCMDLENGTH),.RESULTWIDTH(33),.INITCMDS(I2CINITCMD))
i2cinit(.clk(ethclk)
,.reset(i2cinitreset_x)
,.cmd(i2ccmd)
,.start(i2cinitstart)
,.runing(i2cbusy)//~i2crxvalid)
,.initdone(i2cinitdone[0])
,.dbreset(dbsi2cinitreset)
,.dbstate(dbi2cstate)
,.dbnext(dbi2cnext)
);
assign {lbreg.i2crxvalid[0],lbreg.i2cdatarx}=uartreg.uarti2c ? 0 : i2cresult;
assign {uartreg.i2crxvalid[0],uartreg.i2cdatarx}=uartreg.uarti2c ? i2cresult : 0;
reg [0:0] i2crxvalid_d=0;
reg [0:0] i2crxvalid_d2=0;
reg [31:0] i2cdatarx_d=0;
reg [48+32-1:0] eepromrd=0;
wire [48+32-1:0] eepromrd_x;
alatch #(.DWIDTH(48+32)) eepromrdlatch(.clk(ethclk),.datain(eepromrd),.dataout(eepromrd_x));

reg [3:0] i2cinitdone_d=0;
reg [3:0] i2cinitdone_d2=0;
reg [31:0] i2cdatatx_d=0;
wire [7:0] l=i2cdatatx_d[31:24];
wire [7:0] r={7'h74,1'h1};
wire [7:0] c=i2cdatatx_d[23:16];
wire loadeeprom=(~(i2cinitdone[0]&i2cinitdone_d2[0]) & i2crxvalid_d & ~i2crxvalid_d2 & (i2cswlocation==8) & i2ccmd_rd[24]);
wire updatesw=datatx_run[31:24] =={7'h74,1'h0} & nack_run==2 & stopbit_run;
always @(posedge ethclk) begin
	i2cresult<={i2crxvalid,i2cdatarx};
	i2cinitdone_d<=i2cinitdone;
	i2cinitdone_d2<=i2cinitdone_d;
	i2cdatatx_d<=i2cdatatx;

	{i2crxvalid_d,i2cdatarx_d}<=i2cresult;
	i2crxvalid_d2<=i2crxvalid_d;
	i2ccmd_r<=i2ccmd;
	i2ccmd_rd<=i2ccmd_r;
	i2cbusy_d<=i2cbusy;
	if (loadeeprom) begin
		eepromrd<={eepromrd[80-8-1:0],i2cdatarx_d[7:0]};
	end
	//if (i2cbusy & ~i2cbusy_d & (i2cdatatx_d[31:24]=={7'h74,1'h0})) begin
	if (updatesw) begin
		i2cswlocation<=i2cdatatx_d[23:16];
	end
//	if ((i2ccmd_rd&{1'b1,4'hf,32'hff00ffff})=={1'h1,4'h2,32'he8000000}) begin
	//if ((i2ccmd[31+5:24+5]==8'he8)&&(i2ccmd[15:0]=={16'h0,1'h1,4'h2})) begin
//		i2cswlocation<=i2ccmd_rd[23:16];
//	end

end
assign {lbreg.macmsb24,lbreg.maclsb24,lbreg.ipaddr}=eepromrd;
assign {uartreg.macmsb24,uartreg.maclsb24,uartreg.ipaddr}=eepromrd;
wire fmcdacinitreset_x;
areset fmcdacinitareset(.clk(ethclk),.areset(i2cinitreset[2]),.sreset(fmcdacinitreset_x));
seqinit #(.INITWIDTH(28),.INITLENGTH(INITDACLEN),.INITCMDS(INITDAC))
fmcdacinit(.clk(ethclk)
,.reset(fmcdacinitreset_x)//i2cinitreset[2])
,.cmd(fmci2cdevcmd_2)
,.start(fmci2cstb_devcmd_2)
,.runing(fmci2cbusy)
,.initdone(i2cinitdone[2])
);
wire fmcdac2initreset_x;
areset fmcdac2initareset(.clk(ethclk),.areset(i2cinitreset[3]),.sreset(fmcdac2initreset_x));
seqinit #(.INITWIDTH(28),.INITLENGTH(INITDAC2LEN),.INITCMDS(INITDAC2))
fmcdac2init(.clk(ethclk)
,.reset(fmcdac2initreset_x)//i2cinitreset[3])
,.cmd(fmci2cdevcmd_3)
,.start(fmci2cstb_devcmd_3)
,.runing(fmci2cbusy)
,.initdone(i2cinitdone[3])
);
wire fmclmkadcinitreset_x;
areset fmclmkadcinitareset(.clk(ethclk),.areset(i2cinitreset[1]),.sreset(fmclmkadcinitreset_x));
seqinit #(.INITWIDTH(28),.INITLENGTH(INITLMKADCLEN),.INITCMDS(INITLMKADC))
fmclmkadcinit(.clk(ethclk)
,.reset(fmclmkadcinitreset_x)
,.cmd(fmci2cdevcmd_1)
,.start(fmci2cstb_devcmd_1)
,.runing(fmci2cbusy)
,.initdone(i2cinitdone[1])
);

wire i2creset_x;
areset i2cresetareset(.clk(ethclk),.areset(i2creset),.sreset(i2creset_x));
fmc120cpldspiwriteonvc707 fmc120cpldspiwriteonvc707(.busy(fmci2cbusy)
,.clk(ethclk)
,.reset(i2creset_x)
,.stb_devcmd(fmci2cstb_devcmd)
,.devcmd({fmci2cdevcmd[27:24],4'b0,fmci2cdevcmd[23:0]})
,.running(i2cbusy)
,.nack(fmci2cnack)
,.datatx(fmci2cdatatx)
,.start(fmci2cstart)
,.stopbit(fmci2cstopbit)
,.i2cswlocation(i2cswlocation)
);
assign fmci2ccmd={fmci2cstopbit,fmci2cnack,fmci2cdatatx};
wire sgmiiclk;
IBUFDS_GTE2 mgtrefclk_113_sgmii(.I(hw.vc707.sgmiiclk_q0_p),.IB(hw.vc707.sgmiiclk_q0_n),.O(sgmiiclk),.ODIV2(),.CEB(1'b0));
wire sma_mgt_refclk;
IBUFDS_GTE2 mgtrefclk_113_sma(.I(hw.vc707.sma_mgt_refclk_p),.IB(hw.vc707.sma_mgt_refclk_n),.O(sma_mgt_refclk),.ODIV2(),.CEB(1'b0));
wire si5324_out_c;
IBUFDS_GTE2 mgtrefclk_114_si5324(.I(hw.vc707.si5324_out_c_p),.IB(hw.vc707.si5324_out_c_n),.O(si5324_out_c),.ODIV2(),.CEB(1'b0));
wire pcie_clk_qo;
IBUFDS_GTE2 mgtrefclk1_115_pcie(.I(hw.vc707.pcie.clk_qo_p),.IB(hw.vc707.pcie.clk_qo_n),.O(pcie_clk_qo),.ODIV2());
wire user_clock;
IBUFGDS user_clock_ibufgds(.I(hw.vc707.user_clock_p),.IB(hw.vc707.user_clock_n),.O(user_clock));
wire dspclk;
BUFG dspclkbufg(.I(hw.fmc1.llmk_dclkout_2),.O(dspclk));

wire si5324_out;
wire si5324_out_div2;
reg [2:0] si5324_out_cnt=0;
BUFG si5324bufg(.I(si5324_out_c),.O(si5324_out));
always @(posedge si5324_out_c) begin
	si5324_out_cnt<=si5324_out_cnt+1;
end
//BUFGCE si5324divbufgce(.I(si5324_out_c),.O(si5324_out_div2),.CE(si5324_out_cnt[0]));
wire smamgtclk;
wire smamgtclk_div2;
reg [63:0] smamgtclk_cnt=0;
BUFG smamgtbufg(.I(sma_mgt_refclk),.O(smamgtclk));
always @(posedge sma_mgt_refclk) begin
	smamgtclk_cnt<=smamgtclk_cnt+1;
end
//BUFGCE smamgtdivbufgce(.I(sma_mgt_refclk),.O(smamgtclk_div2),.CE(smamgtclk_cnt[0]));
wire sfpreconnected;
wire reset_sfp_w;
reg hwreset=0;
wire udphwreset;
wire uarthwreset;
areset hwresetareset(.clk(hw.vc707.sysclk),.areset(lbreg.stb_hwreset),.sreset(udphwreset));
areset uarthwresetareset(.clk(hw.vc707.sysclk),.areset(uartreg.stb_hwreset),.sreset(uarthwreset));

//flag_xdomain udphwresetxdomain(.clk1(udplb.clk),.clk2(hw.vc707.sysclk),.flagin_clk1(lbreg.stb_hwreset),.flagout_clk2(udphwreset));
//flag_xdomain uarthwresetxdomain(.clk1(uartclk),.clk2(hw.vc707.sysclk),.flagin_clk1(uartreg.stb_hwreset),.flagout_clk2(uarthwreset));

wire pllresetdonestrobe;
reg sfplos=0;
reg sfplos_d=0;
assign sfpreconnected=~sfplos&sfplos_d;
always @(posedge ethclk) begin
	sfplos<=hw.vc707.sfp.los;
	sfplos_d<=sfplos;
end

iicc #(.DWIDTH(16),.SIM(SIM)) sfpicc();

wire dblocked_sfp;
wire dbrxcdrlock_sfp;
wire [5:0] dbdonecriteria_sfp;
wire  [24:0]  dbdone_sfp;
wire [5:0] dbresetout_sfp;
wire [4:0] dbstate_sfp;
wire [4:0] dbnext_sfp;
wire reset_sfp;
wire stb_txphaseab_cic2;
wire [15:0] txphaseab_cic2;
wire stb_txphaseab_cic;
wire [15:0] txphaseab_cic;
wire [15:0] dbattemptcnt;
wire [4:0] dbtxphase5;
wire [31:0] dbcnt_sfp;
wire txoutclk_sfp;
wire phrefclk62_5;
wire sfpgtreset_x;
areset sfpiccresetareset(.clk(ethclk),.areset(reset_sfp|lbreg.stb_reset_sfp),.sreset(sfpgtreset_x));
gticc_gt #(.SIM_CPLLREFCLK_SEL(3'h002),.DWIDTH(16))
gticc_gt_sfp(.CPLLLOCKDETCLK(ethclk)
//,.GTNORTHREFCLK0(1'b0),.GTNORTHREFCLK1(1'b0),.GTREFCLK0(sgmiiclk),.GTREFCLK1(sma_mgt_refclk),.GTSOUTHREFCLK0(si5324_out_c),.GTSOUTHREFCLK1(1'b0)
,.GTNORTHREFCLK0(1'b0),.GTNORTHREFCLK1(1'b0),.GTREFCLK0(1'b0),.GTREFCLK1(sma_mgt_refclk),.GTSOUTHREFCLK0(1'b0),.GTSOUTHREFCLK1(1'b0)
,.GTXRXN(hw.vc707.sfp.rx_n),.GTXRXP(hw.vc707.sfp.rx_p),.GTXTXN(hw.vc707.sfp.tx_n),.GTXTXP(hw.vc707.sfp.tx_p)
,.QPLLCLK(1'b0),.QPLLREFCLK(1'b0)
,.CPLLREFCLKSEL(3'h2)
,.stb_txphase(stb_txphaseab_cic)
,.txphase(txphaseab_cic)
,.sfplos(sfplos)
,.reset(sfpgtreset_x)
,.txphtarget(lbreg.sfptxphtarget)
,.bypasstxphcheck(lbreg.bypasstxphcheck)
,.gticc(sfpicc.gticc.gt)
,.dblocked(dblocked_sfp)
,.dbrxcdrlock(dbrxcdrlock_sfp)
,.dbdonecriteria(dbdonecriteria_sfp)
//,.mask(lbreg.sfpresetmask)
,.dbdone(dbdone_sfp)
,.dbresetout(dbresetout_sfp)
,.dbstate(dbstate_sfp)
,.dbnext(dbnext_sfp)
,.dbattemptcnt(dbattemptcnt)
,.dbtxphase5(dbtxphase5)
,.dbcnt(dbcnt_sfp)
,.phrefclk62_5(phrefclk62_5)
,.txoutclk(txoutclk_sfp)
);
alatch #(.DWIDTH(10)) dbstatesfplatch(.clk(ethclk),.datain({dbstate_sfp,dbtxphase5}),.dataout({lbreg.dbstate_sfp,lbreg.dbtxphase5}));
reg [4:0] txphase5=0;
wire phasegood=dbtxphase5==lbreg.sfptxphtarget;
wire phasechanged=dbtxphase5==txphase5;
always @(posedge ethclk) begin
	txphase5<=dbtxphase5;

end
assign sfpicc.txdata=lbreg.sfptesttx;
assign sfpicc.stbtxdata=1'b1;
assign lbreg.sfptestrx=sfpicc.rxdata;
assign lbreg.phdiff=sfpicc.phdiff;
wire master;
assign sfpicc.master=master;
wire dmtdreset_w;
reg dmtdreset_r=0;
reg [4:0] ncic=0;
always @(posedge ethclk) begin
	dmtdreset_r<=dmtdreset_w;
	ncic<=SIM ? 5'd3 : lbreg.dmtdnavr;
end
wire [31:0] txphaseab_x;
wire stb_txphaseab_x;
wire [31:0] rxphaseab_x;
wire stb_rxphaseab_x;
wire [31:0] txphaseab_x2;
wire stb_txphaseab_x2;
wire [31:0] rxphaseab_x2;
wire stb_rxphaseab_x2;
simple_cic #(.DWIN(32),.MAXCICW(16),.DWOUT(16))
txphasecic (.clk(ethclk),.reset(dmtdreset_r),.ncic(ncic),.gin(stb_txphaseab_x ),.din(txphaseab_x ),.dout(txphaseab_cic ),.gout(stb_txphaseab_cic ));
simple_cic #(.DWIN(32),.MAXCICW(16),.DWOUT(16))
rxphasecic (.clk(ethclk),.reset(dmtdreset_r),.gin(stb_rxphaseab_x ),.ncic(ncic),.din(rxphaseab_x),.dout(sfpicc.rxphdmtd),.gout());
simple_cic #(.DWIN(32),.MAXCICW(16),.DWOUT(16))
txphasecic2(.clk(ethclk),.reset(dmtdreset_w),.gin(stb_txphaseab_x2),.ncic(ncic),.din(txphaseab_x2),.dout(txphaseab_cic2),.gout(stb_txphaseab_cic2));
simple_cic #(.DWIN(32),.MAXCICW(16),.DWOUT(16))
rxphasecic2(.clk(ethclk),.reset(dmtdreset_w),.gin(stb_rxphaseab_x2),.ncic(ncic),.din(rxphaseab_x2),.dout(smasfpicc.rxphdmtd),.gout());

assign sfpicc.txphdmtd=txphaseab_cic;
assign smasfpicc.txphdmtd=txphaseab_cic2;
assign lbreg.txphaseab=sfpicc.txphdmtd;//txphaseab_x;
assign lbreg.rxphaseab=sfpicc.rxphdmtd;//rxphaseab_x;

//assign sfpicc.rxphdmtd=rxphaseab[15:0];
//assign sfpicc.txphdmtd=txphaseab[15:0];

iicc #(.DWIDTH(16),.SIM(SIM)) smasfpicc();
wire reset_smasfp;//=reset_sfp;
//wire reset_smasfp_w;
//areset resetsmasfpareset(.clk(ethclk),.areset(reset_smasfp),.sreset(reset_smasfp_w));
wire smasfpgtreset_x;
areset smasfpiccresetareset(.clk(ethclk),.areset(reset_smasfp|lbreg.stb_reset_smasfp),.sreset(smasfpgtreset_x));
//assign smasfpicc.sreset=(reset_smasfp_w|lbreg.stb_reset_smasfp);//||smasfpreconnected)
wire dblocked_smasfp;
wire dbrxcdrlock_smasfp;
gticc_gt #(.SIM_CPLLREFCLK_SEL(3'h002),.DWIDTH(16))
gticc_gt_smasfp(.CPLLLOCKDETCLK(ethclk)
//,.GTNORTHREFCLK0(1'b0),.GTNORTHREFCLK1(1'b0),.GTREFCLK0(sgmiiclk),.GTREFCLK1(sma_mgt_refclk),.GTSOUTHREFCLK0(si5324_out_c),.GTSOUTHREFCLK1(1'b0)
,.GTNORTHREFCLK0(1'b0),.GTNORTHREFCLK1(1'b0),.GTREFCLK0(1'b0),.GTREFCLK1(sma_mgt_refclk),.GTSOUTHREFCLK0(1'b0),.GTSOUTHREFCLK1(1'b0)
,.GTXRXN(hw.vc707.sma_mgt_rx_n),.GTXRXP(hw.vc707.sma_mgt_rx_p),.GTXTXN(hw.vc707.sma_mgt_tx_n),.GTXTXP(hw.vc707.sma_mgt_tx_p)
,.QPLLCLK(1'b0),.QPLLREFCLK(1'b0)
,.CPLLREFCLKSEL(3'h2)
,.gticc(smasfpicc.gticc.gt)
,.reset(smasfpgtreset_x)
,.stb_txphase(stb_txphaseab_cic2)
,.txphase(smasfpicc.txphdmtd)
,.txphtarget(lbreg.smasfptxphtarget)
,.bypasstxphcheck(lbreg.bypasstxphcheck)
,.dblocked(dblocked_smasfp)
,.dbrxcdrlock(dbrxcdrlock_smasfp)
);
assign smasfpicc.txdata=lbreg.smasfptesttx;
assign smasfpicc.stbtxdata=1'b1;
assign lbreg.smasfptestrx=smasfpicc.rxdata;//smasfprx;
assign smasfpicc.master=1'b0;
reg [3:0] fmc1dclk10cnt=0;
reg [3:0] fmc2dclk10cnt=0;
reg [3:0] sgmiiclkcnt=0;
always @(posedge hw.fmc1.lmk_dclk10_m2c_to_fpga) begin
	fmc1dclk10cnt<=fmc1dclk10cnt+1;
end
always @(posedge hw.fmc2.lmk_dclk10_m2c_to_fpga) begin
	fmc2dclk10cnt<=fmc2dclk10cnt+1;
end
always @(posedge sgmiiclk) begin
	sgmiiclkcnt<=sgmiiclkcnt+1;
end
reg [63:0] dspclkcnt_orig=0;
always @(posedge dspclk) begin
	dspclkcnt_orig<=dspclkcnt_orig+1;//master ? {smamgtclk_cnt,1'b0} : (dspclkcnt+1);
end
wire [63:0] dspclkcnt=master ? {smamgtclk_cnt,1'b0} : dspclkcnt_orig;

wire [63:0] corr48_ext64;
wire [63:0] corr4_ext64;
sext #(.WIN(4),.WOUT(64)) corr4ext64 (.din(sfpicc.corr4>>>2),.dout(corr4_ext64));
sext #(.WIN(48),.WOUT(64)) corr48ext64(.din(sfpicc.corr48<<<2),.dout(corr48_ext64));
wire [63:0] corr64=corr48_ext64;//+corr4_ext64;
wire [63:0] dspclkcntcorr;
data_xdomain #(.DWIDTH(128)) corrxeth(.clkin(sfpicc.gticc.txusrclk),.clkout(ethclk),.gatein(sfpicc.stb_corr),.datain({sfpicc.cdiff2,sfpicc.pdiff2,sfpicc.p1,sfpicc.p2,sfpicc.p3,sfpicc.p4}),.gateout(),.dataout({lbreg.corrh,lbreg.corrl,lbreg.p12,lbreg.p34}));
assign dspclkcntcorr=dspclkcnt+corr64+lbreg.manualcorr;
reg [63:0] txcnt_r=0;
reg [63:0] txcnt_r1=0;
reg [63:0] txcnt_r2=0;
reg [63:0] txcnt_r3=0;
always @(posedge sfpicc.txclk) begin
	txcnt_r<=(~master & sfpicc.usecorr) ? dspclkcntcorr[49:2] : dspclkcnt[49:2];
	txcnt_r1<=txcnt_r;
	txcnt_r2<=txcnt_r1;
	txcnt_r3<=txcnt_r2;
end
assign sfpicc.txcnt=txcnt_r3;
assign smasfpicc.txcnt=txcnt_r3;
//assign sfpicc.txcnt= dspclkcnt[49:2];
//assign sfpicc.rxcnt=sfpicc.usecorr ? dspclkcntcorr[49:2] : dspclkcnt[49:2];


reg [31:0] fmc2llmkdclkout2cnt=0;
always @(posedge hw.fmc2.llmk_dclkout_2) begin
	fmc2llmkdclkout2cnt<=fmc2llmkdclkout2cnt+1;
end
wire phrefclk125;
assign phrefclk125=master ? smamgtclk : dspclkcnt[0];//hw.fmc1.lmk_dclk10_m2c_to_fpga;
assign phrefclk62_5= master ? smamgtclk_cnt[0] : dspclkcnt[1];

/*wire [32-1:0] dbclkhelpcnt_samp0;
wire [32-1:0] dbclkhelpcnt_samp1;
wire [32-1:0] dbfreqhelp;
wire [32-1:0] dbfreqhelp5;
wire [32-1:0] dbfreqref4;
wire [32-1:0] dbfreqdiff;
wire [32-1:0] dbclkrefcnt;
wire dbsamp;
wire dbsamphelp_d1_ref;
wire dbsamphelp_d1_ref_v;
wire dbsamphelp_v;
wire stb_freqdiff;
wire [37:0] freqdiff;
assign lbreg.freqdiff=freqdiff[31:0];
//helppll45 helppll45err(.clkref(si5324_out_c),.clkhelp(helpclk),.refcntsamp(lbreg.refcntsamp),.freqdiff(freqdiff),.stb_freqdiff(stb_freqdiff)
wire [37:0] refcntsamp;
sext #(.WIN(32),.WOUT(38))
refcntsampsext (.din(lbreg.refcntsamp),.dout(refcntsamp));
helppll45 #(.DWIDTH(38))
helppll45err(.clkref(phrefclk62_5),.clkhelp(helpclk),.refcntsamp(refcntsamp),.freqdiff(freqdiff),.stb_freqdiff(stb_freqdiff)
,.dbclkhelpcnt_samp0(dbclkhelpcnt_samp0)
,.dbclkhelpcnt_samp1(dbclkhelpcnt_samp1)
,.dbfreqhelp(dbfreqhelp)
,.dbfreqhelp5(dbfreqhelp5)
,.dbfreqref4(dbfreqref4)
,.dbfreqdiff(dbfreqdiff)
,.dbsamp(dbsamp)
,.dbsamphelp_d1_ref(dbsamphelp_d1_ref)
,.dbclkrefcnt(dbclkrefcnt)
,.dbsamphelp_d1_ref_v(dbsamphelp_d1_ref_v)
,.dbsamphelp_v(dbsamphelp_v)
);
*/
//wire signed [15:0] freqdiff16;
wire helpclk;
wire signed [31:0] helppllctrl;
wire stb_helppllctrl;
wire helppllpiloopreset=0;
wire [NSTEP-1:0] done_r3;
wire helppllclose=master ? ~|{~i2cinitdone,uartreg.uarti2c,lbreg.lbi2c} : ~|{~i2cinitdone,uartreg.uarti2c,lbreg.lbi2c,~done_r3};
wire [38:0] freqdiff_s;
wire [38:0] freqdiff_sv;
wire [31:0] freqa1;
wire dbavalid;
//areset #(.WIDTH(39)) helppllxdomainethclk (.clk(ethclk),.areset({stb_freqdiff,freqdiff}),.sreset(freqdiff_s),.sreset_val(freqdiff_sv));
always @(posedge ethclk) begin
	if (stb_helppllctrl)
		rfreqfdbk<=$signed(rfreq_now)+$signed(helppllctrl);
	stb_rfreqfdbk<=stb_helppllctrl;
end
//wire stb_freqdiff_x=freqdiff_s[38];
//wire signed [31:0] freqdiff_x;//=freqdiff_sv[37:0];
wire [31:0] dbverr;
//sat #(.WIN(38),.WOUT(32))
//freqdiffsat(.din(freqdiff_sv[37:0]),.dout(freqdiff_x));

wire stb_freqerr;
wire [31:0] freqerr;
wire stb_freqerr_x;
wire [31:0] freqerr_x;
data_xdomain #(.DWIDTH(32)) pherrlatch (.clkin(helpclk),.clkout(ethclk),.gatein(stb_freqerr),.datain(freqerr),.gateout(stb_freqerr_x),.dataout(freqerr_x));
wire helppllpiloopreset_x;
areset helppllpiloopareset(.clk(ethclk),.areset(helppllpiloopreset),.sreset(helppllpiloopreset_x));
piloop5 #(.KPKISHIFTMAX(16),.KISHIFTSTATIC(6),.GWIDTH(16),.DWIDTH(32),.INTEWIDTH(16))
helppllpiloop(.clk(ethclk)
,.reset(helppllpiloopreset_x)
//,.vin(freqdiff_x)
//,.vclose(lbreg.helpplloffset)
//,.stb_vin(stb_freqdiff_x)
//,.vin(freqa1)
//,.stb_vin(dbavalid)
//,.vclose(lbreg.refcntsamp+lbreg.helpplloffset)
,.vin(freqerr_x)
,.stb_vin(stb_freqerr_x)
,.vclose(0)
,.vopen(0)
,.kp(lbreg.helppllkp)
,.kpshift(lbreg.helppllkpshift)
,.ki(lbreg.helppllki)
,.kishift(lbreg.helppllkishift)
,.vout(helppllctrl)
,.closeloop(helppllclose)
,.stb_vout(stb_helppllctrl)
,.dbverr(dbverr)
);


assign dmtdreset_w=|{lbreg.stb_stableval,lbreg.stb_dmtdnavr};
BUFG helpclkbufg(.I(user_clock),.O(helpclk));
wire stb_phdiffavr;
wire [31:0] phdiffavr;
wire [31:0] phdiffmidavr;
wire signed [31:0] dbafreq;
wire signed [31:0] dbbfreq;
wire [31:0] dbastable,dbbstable,dbphdiff,dbacc1;
wire dbpvalid;
wire dbsclka;
wire dbsclkb;
wire [31:0] dbclkdmtdcnt;
wire dbbvalid;
wire [1:0] dbdmtdstate;
wire [1:0] dbdmtdnext;
wire [2:0] dbastate;
wire [2:0] dbanext;
wire dbaval;
wire [2:0] dbbstate;
wire [2:0] dbbnext;
wire dbbval;
wire [31:0] dbsclkacnt;
wire [31:0] dbsclkbcnt;
wire dbstable_sclka;
wire dbstable_sclkb;
wire phsrc;
assign phsrc=	lbreg.phsrc==0 ? fmc1dclk10cnt[0]:
	lbreg.phsrc==1 ? fmc2dclk10cnt[0] :
	lbreg.phsrc==2 ? si5324_out_cnt[0] :
	lbreg.phsrc==3 ? smamgtclk_cnt[0] :
	lbreg.phsrc==4 ? sfpicc.gticc.txusrclk : //_sfp :
	lbreg.phsrc==5 ? sfpicc.gticc.rxusrclk : //_sfp :
//	lbreg.phsrc==6 ? smasfpicc.gticc.txusrclk : //_smasfp :
//	lbreg.phsrc==7 ? smasfpicc.gticc.rxusrclk :
	0 ; //_smasfp : 0;
wire [31:0] freqa;
wire [31:0] freqb;
wire [31:0] dbfreecnta;
wire [31:0] dbfreecntb;
wire [31:0] phaseab;
wire stb_phaseab;
/*dmtd #(.SIM(SIM),.FOFFSETWIDTH(SIM ? 9 : 14))
dmtdtest (.clkdmtd(helpclk),.rst(dmtdreset_w)
//,.navr(lbreg.dmtdnavr)
,.stableval(lbreg.stableval)
,.clka(phrefclk62_5),.clkb(phsrc)
//,.clka(si5324_out_cnt[0]),.clkb(smamgtclk_cnt[0])
//,.clka(rxusrclk_sfp),.clkb(rxusrclk_smasfp)
//,.phdiffavr(phdiffavr),.stb_phdiffavr(stb_phdiffavr)
//,.phdiffmidavr(phdiffmidavr)
,.freqerr()
,.stb_freqerr()
,.stb_phaseab(stb_phaseab)
,.phaseab(phaseab)

);
*/
wire dmtdreset_x;
areset dmtdareset(.clk(helpclk),.areset(dmtdreset_w),.sreset(dmtdreset_x));
wire [31:0] txphaseab;
wire [31:0] rxphaseab;
wire stb_txphaseab;
wire stb_rxphaseab;
wire dbafreq01;
wire [31:0] dbfreqint;
wire [31:0] dbfracerr;
dmtd #(.SIM(SIM),.FOFFSETWIDTH(SIM ? 9 : 14))
dmtdtx (.clkdmtd(helpclk),.rst(dmtdreset_x),.stableval(SIM ? 10 : lbreg.stableval)
//,.clka(phrefclk62_5),.clkb(sfpicc.gticc.txusrclk)
,.clka(phrefclk62_5),.clkb(txoutclk_sfp)//sfpicc.gticc.txusrclk)
,.offsetwidth(lbreg.offsetwidth)
,.freqerr(freqerr)
,.stb_freqerr(stb_freqerr)
,.stb_phaseab(stb_txphaseab)
,.phaseab(txphaseab)

,.dbfreqint(dbfreqint),.dbfracerr(dbfracerr)
,.freqa(freqa),.freqb(freqb)
,.dbastable(dbastable)
,.dbbstable(dbbstable)
,.dbphdiff(dbphdiff)
,.dbafreq(dbafreq)
,.dbbfreq(dbbfreq)
,.dbsclka(dbsclka)
,.dbsclkb(dbsclkb)
,.dbclkdmtdcnt(dbclkdmtdcnt)
,.dbavalid(dbavalid)
,.dbbvalid(dbbvalid)
,.dbstate(dbdmtdstate)
,.dbnext(dbdmtdnext)
,.dbsclkacnt(dbsclkacnt)
,.dbsclkbcnt(dbsclkbcnt)
,.dbstable_sclka(dbstable_sclka)
,.dbstable_sclkb(dbstable_sclkb)
,.dbfreecnta(dbfreecnta)
,.dbfreecntb(dbfreecntb)
);
dmtd #(.SIM(SIM),.FOFFSETWIDTH(SIM ? 9 : 14))
dmtdrx (.clkdmtd(helpclk),.rst(dmtdreset_x),.stableval(SIM ? 10 : lbreg.stableval)
,.clka(phrefclk62_5),.clkb(sfpicc.gticc.rxusrclk)
,.freqerr()
,.stb_freqerr()
,.stb_phaseab(stb_rxphaseab)
,.phaseab(rxphaseab)
);
wire [32*4+1-1:0] dmtd_x;
wire [32*4+1-1:0] dmtd_xval;
reg [32*4+1-1:0] dmtd_xval_r=0;
wire [31:0] freqa_x;
wire [31:0] freqb_x;
wire [31:0] phdiffavr_x;
wire [31:0] phaseab_x;
wire [31:0] phdiffmidavr_x;

//data_xdomain #(.DWIDTH(32*2))dmtdhelpclktoethclk(.clkin(helpclk),.clkout(ethclk),.gatein(stb_phaseab),.datain({freqa,freqb}),.gateout(),.dataout({lbreg.phaseab,lbreg.freqa,lbreg.freqb}));
data_xdomain #(.DWIDTH(32*3))dmtdtxlatch(.clkin(helpclk),.clkout(ethclk),.gatein(stb_txphaseab),.datain({txphaseab,freqa,freqb}),.gateout(stb_txphaseab_x),.dataout({txphaseab_x,lbreg.freqa,lbreg.freqb}));
data_xdomain #(.DWIDTH(32))dmtdrxlatch(.clkin(helpclk),.clkout(ethclk),.gatein(stb_rxphaseab),.datain(rxphaseab),.gateout(stb_rxphaseab_x),.dataout(rxphaseab_x));



wire [31:0] txphaseab2;
wire [31:0] rxphaseab2;
wire stb_txphaseab2;
wire stb_rxphaseab2;
dmtd #(.SIM(SIM),.FOFFSETWIDTH(SIM ? 9 : 14))
dmtdtx2(.clkdmtd(helpclk),.rst(dmtdreset_x),.stableval(SIM ? 10 : lbreg.stableval)
,.clka(phrefclk62_5),.clkb(smasfpicc.gticc.txusrclk)
,.offsetwidth(lbreg.offsetwidth)
,.freqerr()
,.stb_freqerr()
,.stb_phaseab(stb_txphaseab2)
,.phaseab(txphaseab2)
);
dmtd #(.SIM(SIM),.FOFFSETWIDTH(SIM ? 9 : 14))
dmtdrx2(.clkdmtd(helpclk),.rst(dmtdreset_x),.stableval(SIM ? 10 : lbreg.stableval)
,.clka(phrefclk62_5),.clkb(smasfpicc.gticc.rxusrclk)
,.freqerr()
,.stb_freqerr()
,.stb_phaseab(stb_rxphaseab2)
,.phaseab(rxphaseab2)
);
wire [32*4+1-1:0] dmtd_x2;
wire [32*4+1-1:0] dmtd_xval2;
reg [32*4+1-1:0] dmtd_xval_r2=0;

//data_xdomain #(.DWIDTH(32*2))dmtdhelpclktoethclk(.clkin(helpclk),.clkout(ethclk),.gatein(stb_phaseab),.datain({freqa,freqb}),.gateout(),.dataout({lbreg.phaseab,lbreg.freqa,lbreg.freqb}));
data_xdomain #(.DWIDTH(32))dmtdtxlatch2(.clkin(helpclk),.clkout(ethclk),.gatein(stb_txphaseab2),.datain(txphaseab2),.gateout(stb_txphaseab_x2),.dataout(txphaseab_x2));
data_xdomain #(.DWIDTH(32))dmtdrxlatch2(.clkin(helpclk),.clkout(ethclk),.gatein(stb_rxphaseab2),.datain(rxphaseab2),.gateout(stb_rxphaseab_x2),.dataout(rxphaseab_x2));




wire sgmii_ethernet_pcs_pma_mdc;
wire sgmii_ethernet_pcs_pma_mdio_i;//=1'b1;// don't know how to use yet
wire sgmii_ethernet_pcs_pma_mdio_o;
wire sgmii_ethernet_pcs_pma_mdio_t;
wire [15:0] status_vector;
wire sgmiieth_reset;
wire sgmiieth_resetdone;
wire sgmiieth_reset_x;
areset sgmiiareset (.clk(clk125),.areset(sgmiieth_reset),.sreset(sgmiieth_reset_x));
gmii gmii();
sgmii_ethernet_pcs_pma #(.SIM(SIM),.PHYADDR(5'b00110))
sgmii_ethernet_pcs_pma(.gtrefclk(sgmiiclk)
,.rxn(hw.vc707.sgmii_rx_n)
,.rxp(hw.vc707.sgmii_rx_p)
,.txn(hw.vc707.sgmii_tx_n)
,.txp(hw.vc707.sgmii_tx_p)
,.gmii(gmii.phy)
,.independent_clock_bufg(clk125)
,.reset(sgmiieth_reset_x)  // async
,.resetdone(sgmiieth_resetdone)
,.status_vector(status_vector)

,.mdc(sgmii_ethernet_pcs_pma_mdc)
,.mdio_i(sgmii_ethernet_pcs_pma_mdio_i)
,.mdio_o(sgmii_ethernet_pcs_pma_mdio_o)
,.mdio_t(sgmii_ethernet_pcs_pma_mdio_t)
);

//assign ethclk=clk125;//gmii.tx_clk;

wire mdioreset;
wire mdioinitdone;
wire ethreset_w;
wire ethreset;
areset ethareset(.clk(ethclk),.areset(ethreset_w),.sreset(ethreset));
wire ethresetdone=~ethreset;
wire jesdreset0_done;
wire jesdreset1_done;
wire jesd_reset_done_1;
wire jesd_reset_done_2;
wire jesdreset0;
wire jesdreset1;
wire axireset_sys;
wire axireset;
wire axiinitreset;
wire axiinitdone;
wire loadmacip;

areset axiareset(.clk(ethclk),.areset(axireset_sys),.sreset(axireset));

wire  mdio_i;
wire  mdio_o;
wire  mdio_t;
wire  phy_mdio_i;
wire  phy_mdio_o=mdio_o;
wire  phy_mdio_t=mdio_t;
IOBUF mdiobuf(.O(phy_mdio_i),.I(phy_mdio_o),.T(phy_mdio_t),.IO(hw.vc707.phy_mdio));
wire  opr1w0;
wire [4:0] phyaddr;
wire [4:0] regaddr;
wire [15:0] datatx;
reg stb_mdiostart_d=0;
reg [31:0] mdioclk4ratio=0;
reg [26:0] mdiodatatx=0;
wire [31:0] mdiodatarx;
wire mdiorxvalid;
assign lbreg.mdiodatarx=mdiodatarx;
assign uartreg.mdiodatarx=mdiodatarx;
assign lbreg.mdiorxvalid=mdiorxvalid;
assign uartreg.mdiorxvalid=mdiorxvalid;
reg mdioreset_r=1'b0;
wire mdioreset_x;
wire mdc;
reg phy_mdc_r=0;
wire phy_reset;
reg [31:0] phy_resetwait_cnt=0;
wire phy_resetwait;
wire phy_resetwait_done=(phy_resetwait_cnt==(SIM ? 32'h80 : 32'h800000));
always @(posedge hw.vc707.sysclk) begin
	phy_resetwait_cnt<=phy_resetwait ? 0 : (phy_resetwait_cnt+ (phy_resetwait_done ? 1'b0: 1'b1));
end
reg phy_reset_r=0;
wire phy_reset_x;
alatch #(.DWIDTH(1)) phyresetxdomain(.clk(ethclk),.datain(phy_reset),.dataout(phy_reset_x));
always @(posedge ethclk) begin
	if (lbreg.stb_mdiostart) begin
		mdioclk4ratio<=lbreg.mdioclk4ratio ;
		stb_mdiostart_d<=lbreg.stb_mdiostart ;
		mdiodatatx<=lbreg.mdiodatatx[26:0] ;
	end
	else if (uartreg.stb_mdiostart) begin
		mdioclk4ratio<=uartreg.mdioclk4ratio;
		stb_mdiostart_d<=uartreg.stb_mdiostart;
		mdiodatatx<= uartreg.mdiodatatx[26:0];
	end
	else begin
		stb_mdiostart_d<=0;
		mdiodatatx<=0;
	end
	phy_reset_r<=phy_reset_x; // the task before mdioreset
	phy_mdc_r<=mdc;
end
assign {opr1w0,phyaddr,regaddr,datatx}=mdiodatatx;
assign hw.vc707.phy_reset=~phy_reset;// the task before mdioreset
assign hw.vc707.phy_mdc=mdc;
wire phy_int=hw.vc707.phy_int;
areset mdioresetareset (.clk(ethclk),.areset(mdioreset),.sreset(mdioreset_x));
mdiomasterinit #(.INITLENGTH(5),.INITWIDTH(27),.INITCLK4RATIO(SIM ? 2 : 100),.INITCMDS(
    {
    1'b0,5'b00110,5'h0,16'h0140
	,1'b0,5'b00111,5'h0,16'h0140
    ,1'b0,5'b00111,5'h4,16'h9801
    ,1'b0,5'b00111,5'h16,16'h1
    ,1'b0,5'b00111,5'h0,16'h8140
    //,1'b0,5'b00110,5'h0,16'h8140
    }
))
mdiomasterinit(.clk(ethclk),.busy(),.clk4ratio(SIM ? 32'd10 : mdioclk4ratio),.datarx(mdiodatarx),.datatx(datatx),.mdc(mdc),.mdio_i(mdio_i),.mdio_o(mdio_o),.mdio_t(mdio_t),.opr1w0(opr1w0),.phyaddr(phyaddr),.regaddr(regaddr),.rst(phy_reset_x),.rxvalid(mdiorxvalid),.start(stb_mdiostart_d),.initdone(mdioinitdone),.initstart(mdioreset_x));

assign sgmii_ethernet_pcs_pma_mdc=mdc;
assign mdio_i=&{sgmii_ethernet_pcs_pma_mdio_o,phy_mdio_i};
assign sgmii_ethernet_pcs_pma_mdio_i=mdio_o;
//assign sgmii_ethernet_pcs_pma_mdio_t=mdio_t;

wire [7:0] last_ip_byte=8'b0;
wire [23:0] lb_addr;
wire [31:0] lb_data_in=32'hdeadbeef;
wire [31:0] lb_data_out;
wire  lb_read;
wire  lb_rvalid;
wire  lb_write;
wire  pwm_out0;
wire  pwm_out1;
wire  reset=1'b0;
wire [7:0] s_tx_tdata=8'b0;
wire  s_tx_tready;
wire  s_tx_tvalid=1'b0;
wire [7:0] status;

//reg [47:0] mac=48'h503eaa059701;
reg [47:0] mac=48'h515542494301;
assign keeplbdataout=&lb_data_out;
reg [31:0] ip=32'hc0a801e0;
assign ethclk=gmii.tx_clk;
reg maciploaded=1'b0;
wire loadmacip_s;
areset loadmacipreset(.clk(ethclk),.areset(loadmacip),.sreset(loadmacip_s));
always @(posedge ethclk) begin
//	mac<=48'h525542494301;//MAC;//48'haabbccddeeff;
//	mac<=48'h001924515501;  // LBNL oui
	if (loadmacip_s) begin
		//{mac,ip}<=hw.vc707.gpio_sw_c ? {48'h503eaa059701,32'hc0a801e0} : eepromrd;  // TPLINK OUI
		{mac,ip}<=1'b1 ? {48'h503eaa059701,32'hc0a801e0} : eepromrd;  // TPLINK OUI
		maciploaded<=1'b1;
	end

//	ip<=hw.gpio_sw_c ? : ;  // 192.168.1.224
// new:	50:3e:aa:05:96:50
end
udplink ifudp(.reset(ethreset),.clk(ethclk));
gmii2udp #(.DEBUG(DEBUG),.SIM(SIM))
gmii2udp(.gmii(gmii.eth),.ifudp(ifudp),.mac(mac),.ip(ip),.reset(ethreset));
wire [63:0] udprxerr;
udplink ifudpportd001(.reset(ethreset),.clk(ethclk));
udplink ifudpportd002(.reset(ethreset),.clk(ethclk));
udplink ifudpportd003(.reset(ethreset),.clk(ethclk));
udplink ifudpportd000(.reset(ethreset),.clk(ethclk));
udpsw udpsw(.udp(ifudp),.udpportd001(ifudpportd001),.udpportd000(ifudpportd000),.udpportd002(ifudpportd002),.udpportd003(ifudpportd003));
udpecho #(.PORT(16'hd000))
udpecho(.clk(ethclk),.udp(ifudpportd000),.reset(ethreset));
udpstatic #(.PORT(16'hd001))
udpstatic(.clk(ethclk),.udp(ifudpportd001),.reset(ethreset),.staticnbyte(16'd1472));
udpcnt #(.PORT(16'hd002))
udpcnt(.clk(ethclk),.udp(ifudpportd002),.reset(ethreset),.udprxerr(udprxerr),.countperrequest(lbreg.countperrequest));
wire [15:0] txlength;
wire [15:0] rxlength;
udplb64 #(.PORT(16'hd003))
udplb64 (.clk(ethclk),.udp(ifudpportd003),.reset(ethreset)//~sgmiieth_resetdone)
,.lbclk(udplb.clk)
,.lbrxdata(udplb.wcmd)
,.lbrxdv(udplb.wvalid)
,.lbtxdata(udplb.rcmd)
,.lbtxen(udplb.rready)
,.lbrxen(udplb.wen)
,.rxlength(rxlength)
,.txlength(txlength)
);
assign udplb.clk=ethclk;
assign txlength=rxlength;
//assign lbreg.lbrready=lbreg.lbwvalid;  // for this current uart lb, response immidiately
wire [31:0] test=lbreg.test;

///*
wire [63:0] adc0;
wire [63:0] adc1;
wire [63:0] adc2;
wire [63:0] adc3;
wire [63:0] adc4;
wire [63:0] adc5;
wire [63:0] adc6;
wire [63:0] adc7;
wire [63:0] dac0;
wire [63:0] dac1;
wire [63:0] dac2;
wire [63:0] dac3;
wire [63:0] dac4;
wire [63:0] dac5;
wire [63:0] dac6;
wire [63:0] dac7;
assign keepadc=|{adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7};
wire [2:0] dbaxistate;
wire [2:0] dbaxinext;
axi4lite axi_fmc1_adc0(.aclk(ethclk));
axi4lite axi_fmc1_adc1(.aclk(ethclk));
axi4lite axi_fmc1_dac(.aclk(ethclk));
axi4lite axi_fmc2_adc0(.aclk(ethclk));
axi4lite axi_fmc2_adc1(.aclk(ethclk));
axi4lite axi_fmc2_dac(.aclk(ethclk));
reg [11:0] fmc1_adc0_addr=0,fmc1_adc1_addr=0,fmc2_adc0_addr=0,fmc2_adc1_addr=0,fmc1_dac_addr=0,fmc2_dac_addr=0;
reg [31:0] fmc1_adc0_wdata=0,fmc1_adc1_wdata=0,fmc2_adc0_wdata=0,fmc2_adc1_wdata=0,fmc1_dac_wdata=0,fmc2_dac_wdata=0;
reg [3:0] fmc1_adc0_wstrb=0,fmc1_adc1_wstrb=0,fmc2_adc0_wstrb=0,fmc2_adc1_wstrb=0,fmc1_dac_wstrb=0,fmc2_dac_wstrb=0;
reg fmc1_adc0_start=0,fmc1_adc1_start=0,fmc2_adc0_start=0,fmc2_adc1_start=0,fmc1_dac_start=0,fmc2_dac_start=0;
reg fmc1_adc0_w0r1=0,fmc1_adc1_w0r1=0,fmc2_adc0_w0r1=0,fmc2_adc1_w0r1=0,fmc1_dac_w0r1=0,fmc2_dac_w0r1=0;
wire [46:0] axiinitcmd;
wire [5:0] axibusy;
`ifdef JESD
wire axiinitreset_x;
areset axiinitareset(.clk(ethclk),.areset(axiinitreset),.sreset(axiinitreset_x));
seqinit #(.INITWIDTH(3+12+32),.INITLENGTH(AXIINITCMDLENGTH),.INITCMDS(AXIINITCMD))
axiinit(.clk(ethclk)
,.reset(axiinitreset_x)
,.cmd(axiinitcmd)
,.start(axiinitstart)
,.runing(|axibusy)
,.initdone(axiinitdone)
);
wire [2:0] axiid;
always @(posedge ethclk) begin
{fmc1_adc0_addr,fmc1_adc0_wdata,fmc1_adc0_wstrb,fmc1_adc0_w0r1,fmc1_adc0_start}=(~axiinitdone & axiinitcmd[46:44]==3'h0) ? {axiinitcmd[43:0],4'hf,1'b0,axiinitstart} : {lbreg.axifmc1adc0_addr,lbreg.axifmc1adc0_wdata ,4'hf,lbreg.axifmc1adc0_w0r1,lbreg.stb_axifmc1adc0_start};
{fmc1_adc1_addr,fmc1_adc1_wdata,fmc1_adc1_wstrb,fmc1_adc1_w0r1,fmc1_adc1_start}=(~axiinitdone & axiinitcmd[46:44]==3'h1) ? {axiinitcmd[43:0],4'hf,1'b0,axiinitstart} : {lbreg.axifmc1adc1_addr,lbreg.axifmc1adc1_wdata ,4'hf,lbreg.axifmc1adc1_w0r1,lbreg.stb_axifmc1adc1_start};
{fmc1_dac_addr,fmc1_dac_wdata,fmc1_dac_wstrb,fmc1_dac_w0r1,fmc1_dac_start}=(~axiinitdone & axiinitcmd[46:44]==3'h2) ? {axiinitcmd[43:0],4'hf,1'b0,axiinitstart} : {lbreg.axifmc1dac_addr,lbreg.axifmc1dac_wdata ,4'hf,lbreg.axifmc1dac_w0r1,lbreg.stb_axifmc1dac_start};
{fmc2_adc0_addr,fmc2_adc0_wdata,fmc2_adc0_wstrb,fmc2_adc0_w0r1,fmc2_adc0_start}=(~axiinitdone & axiinitcmd[46:44]==3'h3) ? {axiinitcmd[43:0],4'hf,1'b0,axiinitstart} : {lbreg.axifmc2adc0_addr,lbreg.axifmc2adc0_wdata ,4'hf,lbreg.axifmc2adc0_w0r1,lbreg.stb_axifmc2adc0_start};
{fmc2_adc1_addr,fmc2_adc1_wdata,fmc2_adc1_wstrb,fmc2_adc1_w0r1,fmc2_adc1_start}=(~axiinitdone & axiinitcmd[46:44]==3'h4) ? {axiinitcmd[43:0],4'hf,1'b0,axiinitstart} : {lbreg.axifmc2adc1_addr,lbreg.axifmc2adc1_wdata ,4'hf,lbreg.axifmc2adc1_w0r1,lbreg.stb_axifmc2adc1_start};
{fmc2_dac_addr,fmc2_dac_wdata,fmc2_dac_wstrb,fmc2_dac_w0r1,fmc2_dac_start}=(~axiinitdone & axiinitcmd[46:44]==3'h5) ? {axiinitcmd[43:0],4'hf,1'b0,axiinitstart} : {lbreg.axifmc2dac_addr,lbreg.axifmc2dac_wdata ,4'hf,lbreg.axifmc2dac_w0r1,lbreg.stb_axifmc2dac_start};
end

lb_axi4lite #(.AWIDTH(12),.DWIDTH(32))
lb_axi4lite_fmc1_adc0
(.clk(ethclk),.slave(axi_fmc1_adc0),.addr(fmc1_adc0_addr),.wdata(fmc1_adc0_wdata),.wstrb(fmc1_adc0_wstrb),.rdata(lbreg.axifmc1adc0_rdata),.rdatavalid(lbreg.axifmc1adc0_rdatavalid),.start(fmc1_adc0_start),.w0r1(fmc1_adc0_w0r1),.reset(axireset),.dbstate(dbaxistate),.dbnext(dbaxinext),.busy(axibusy[0]));
lb_axi4lite #(.AWIDTH(12),.DWIDTH(32))
lb_axi4lite_fmc1_adc1
(.clk(ethclk),.slave(axi_fmc1_adc1),.addr(fmc1_adc1_addr),.wdata(fmc1_adc1_wdata),.wstrb(fmc1_adc1_wstrb),.rdata(lbreg.axifmc1adc1_rdata),.rdatavalid(lbreg.axifmc1adc1_rdatavalid),.start(fmc1_adc1_start),.w0r1(fmc1_adc1_w0r1),.reset(axireset),.dbstate(),.dbnext(),.busy(axibusy[1]));
lb_axi4lite #(.AWIDTH(12),.DWIDTH(32))
lb_axi4lite_fmc1_dac
(.clk(ethclk),.slave(axi_fmc1_dac),.addr(fmc1_dac_addr),.wdata(fmc1_dac_wdata),.wstrb(fmc1_dac_wstrb),.rdata(lbreg.axifmc1dac_rdata),.rdatavalid(lbreg.axifmc1dac_rdatavalid),.start(fmc1_dac_start),.w0r1(fmc1_dac_w0r1),.reset(axireset),.dbstate(),.dbnext(),.busy(axibusy[2]));
lb_axi4lite #(.AWIDTH(12),.DWIDTH(32))
lb_axi4lite_fmc2_adc0
(.clk(ethclk),.slave(axi_fmc2_adc0),.addr(fmc2_adc0_addr),.wdata(fmc2_adc0_wdata),.wstrb(fmc2_adc0_wstrb),.rdata(lbreg.axifmc2adc0_rdata),.rdatavalid(lbreg.axifmc2adc0_rdatavalid),.start(fmc2_adc0_start),.w0r1(fmc2_adc0_w0r1),.reset(axireset),.dbstate(),.dbnext(),.busy(axibusy[3]));
lb_axi4lite #(.AWIDTH(12),.DWIDTH(32))
lb_axi4lite_fmc2_adc1
(.clk(ethclk),.slave(axi_fmc2_adc1),.addr(fmc2_adc1_addr),.wdata(fmc2_adc1_wdata),.wstrb(fmc2_adc1_wstrb),.rdata(lbreg.axifmc2adc1_rdata),.rdatavalid(lbreg.axifmc2adc1_rdatavalid),.start(fmc2_adc1_start),.w0r1(fmc2_adc1_w0r1),.reset(axireset),.dbstate(),.dbnext(),.busy(axibusy[4]));
lb_axi4lite #(.AWIDTH(12),.DWIDTH(32))
lb_axi4lite_fmc2_dac
(.clk(ethclk),.slave(axi_fmc2_dac),.addr(fmc2_dac_addr),.wdata(fmc2_dac_wdata),.wstrb(fmc2_dac_wstrb),.rdata(lbreg.axifmc2dac_rdata),.rdatavalid(lbreg.axifmc2dac_rdatavalid),.start(fmc2_dac_start),.w0r1(fmc2_dac_w0r1),.reset(axireset),.dbstate(),.dbnext(),.busy(axibusy[5]));
`else
assign axiinitdone=1'b1;
`endif
//(.clk(ethclk),.slave(axi_fmc1_adc1),.addr(lbreg.axifmc1adc1_addr),.wdata(lbreg.axifmc1adc1_wdata),.wstrb(4'hf),.rdata(lbreg.axifmc1adc1_rdata),.rdatavalid(lbreg.axifmc1adc1_rdatavalid),.start(lbreg.stb_axifmc1adc1_start),.w0r1(lbreg.axifmc1adc1_w0r1),.reset(axireset));
wire [1:0] jesd_reset_status_1;
wire [1:0] jesd_reset_status_2;
wire adc01_valid;
wire adc23_valid;
wire fmc1_tx_tready;
wire fmc2_tx_tready;
wire common0_qpll_lock_out_1;
wire common1_qpll_lock_out_2;
wire rxencommaalign_0;
wire rxencommaalign_1;
reg [1:0] rxencommaalign;
always @(posedge ethclk) begin
	rxencommaalign<={rxencommaalign_0,rxencommaalign_1};
end
reg [31:0] dbgt0_rxdata_r=0, dbgt0_txdata_r=0, dbgt1_rxdata_r=0, dbgt1_txdata_r=0, dbgt2_rxdata_r=0, dbgt2_txdata_r=0, dbgt3_rxdata_r=0, dbgt3_txdata_r=0, dbgt4_rxdata_r=0, dbgt4_txdata_r=0, dbgt5_rxdata_r=0, dbgt5_txdata_r=0, dbgt6_rxdata_r=0, dbgt6_txdata_r=0, dbgt7_rxdata_r=0, dbgt7_txdata_r=0;
wire [31:0] dbgt0_rxdata, dbgt0_txdata, dbgt1_rxdata, dbgt1_txdata, dbgt2_rxdata, dbgt2_txdata, dbgt3_rxdata, dbgt3_txdata, dbgt4_rxdata, dbgt4_txdata, dbgt5_rxdata, dbgt5_txdata, dbgt6_rxdata, dbgt6_txdata, dbgt7_rxdata, dbgt7_txdata;
always @(posedge dspclk) begin
	{dbgt0_rxdata_r, dbgt0_txdata_r, dbgt1_rxdata_r, dbgt1_txdata_r, dbgt2_rxdata_r, dbgt2_txdata_r, dbgt3_rxdata_r, dbgt3_txdata_r, dbgt4_rxdata_r, dbgt4_txdata_r, dbgt5_rxdata_r, dbgt5_txdata_r, dbgt6_rxdata_r, dbgt6_txdata_r, dbgt7_rxdata_r, dbgt7_txdata_r}<={dbgt0_rxdata, dbgt0_txdata, dbgt1_rxdata, dbgt1_txdata, dbgt2_rxdata, dbgt2_txdata, dbgt3_rxdata, dbgt3_txdata, dbgt4_rxdata, dbgt4_txdata, dbgt5_rxdata, dbgt5_txdata, dbgt6_rxdata, dbgt6_txdata, dbgt7_rxdata, dbgt7_txdata};
end
wire [1:0] rx_sync_1;
wire [1:0] rx_sync_2;
reg [1:0] rx_sync_1_d=0;
reg [1:0] rx_sync_2_d=0;
always @(posedge dspclk) begin
	rx_sync_1_d<=rx_sync_1;
	rx_sync_2_d<=rx_sync_2;
end
assign {hw.fmc1.adca_sync_in_l_vadj,hw.fmc1.adcb_sync_in_l_vadj}=rx_sync_1;
assign {hw.fmc2.adca_sync_in_l_vadj,hw.fmc2.adcb_sync_in_l_vadj}=rx_sync_2;
wire jesdreset1_x;
wire jesdreset0_x;
wire rx_reset=jesdreset1_x;//1'b0;
wire tx_reset=jesdreset1_x;//1'b0;
wire rx_sys_reset=jesdreset0_x;//1'b0;
wire tx_sys_reset=jesdreset0_x;//1'b0;
areset jesdreset1areset(.clk(ethclk),.areset(jesdreset1),.sreset(jesdreset1_x));
areset jesdreset0areset(.clk(ethclk),.areset(jesdreset0),.sreset(jesdreset0_x));
//generate
//if (SIM==0) begin
`ifdef JESD
jesdfmc120 jesdfmc120_1(.core_clk(dspclk)
,.drpclk(ethclk)
,.qpll_refclk(hw.fmc1.lmk_dclk8_m2c_to_fpga)
,.rxn_in({hw.fmc1.adc1_db2_n,hw.fmc1.adc1_db1_n,hw.fmc1.adc1_da2_n,hw.fmc1.adc1_da1_n,hw.fmc1.adc0_db2_n,hw.fmc1.adc0_db1_n,hw.fmc1.adc0_da2_n,hw.fmc1.adc0_da1_n})
,.rxp_in({hw.fmc1.adc1_db2_p,hw.fmc1.adc1_db1_p,hw.fmc1.adc1_da2_p,hw.fmc1.adc1_da1_p,hw.fmc1.adc0_db2_p,hw.fmc1.adc0_db1_p,hw.fmc1.adc0_da2_p,hw.fmc1.adc0_da1_p})
,.tx_sysref(hw.fmc1.llmk_sclkout_3),.rx_sysref(hw.fmc1.llmk_sclkout_3),.txn_out(hw.fmc1.dac_lane_n),.txp_out(hw.fmc1.dac_lane_p),.axi_adc0(axi_fmc1_adc0),.axi_adc1(axi_fmc1_adc1),.axi_dac(axi_fmc1_dac),.rx_reset(rx_reset),.tx_reset(tx_reset),.rx_sys_reset(rx_sys_reset),.tx_sys_reset(tx_sys_reset),.rx_sync(rx_sync_1),.tx_sync(hw.fmc1.dac_sync_req_to_fpga),.adc0(adc0),.adc1(adc1),.adc2(adc2),.adc3(adc3),.dac0(dac0),.dac1(dac1),.dac2(dac2),.dac3(dac3),.rx_aresetn_0(),.rx_aresetn_1(),.rx_frame_error(),.tx_aresetn(),.tx_tready(fmc1_tx_tready),.adc01_valid(adc01_valid),.adc23_valid(adc23_valid),.common0_qpll_lock_out(common0_qpll_lock_out_1),.common1_qpll_lock_out(common1_qpll_lock_out_2),.reset_done(jesd_reset_done_1),.reset_status(jesd_reset_status_1),.rxencommaalign_0(rxencommaalign_0),.rxencommaalign_1(rxencommaalign_1)
,.dbgt0_rxdata,.dbgt0_txdata,.dbgt1_rxdata,.dbgt1_txdata,.dbgt2_rxdata,.dbgt2_txdata,.dbgt3_rxdata,.dbgt3_txdata,.dbgt4_rxdata,.dbgt4_txdata,.dbgt5_rxdata,.dbgt5_txdata,.dbgt6_rxdata,.dbgt6_txdata,.dbgt7_rxdata,.dbgt7_txdata
);
jesdfmc120 jesdfmc120_2(.core_clk(dspclk)
,.drpclk(ethclk)
,.qpll_refclk(hw.fmc2.lmk_dclk8_m2c_to_fpga)
,.rxn_in({hw.fmc2.adc1_db2_n,hw.fmc2.adc1_db1_n,hw.fmc2.adc1_da2_n,hw.fmc2.adc1_da1_n,hw.fmc2.adc0_db2_n,hw.fmc2.adc0_db1_n,hw.fmc2.adc0_da2_n,hw.fmc2.adc0_da1_n})
,.rxp_in({hw.fmc2.adc1_db2_p,hw.fmc2.adc1_db1_p,hw.fmc2.adc1_da2_p,hw.fmc2.adc1_da1_p,hw.fmc2.adc0_db2_p,hw.fmc2.adc0_db1_p,hw.fmc2.adc0_da2_p,hw.fmc2.adc0_da1_p})
,.tx_sysref(hw.fmc2.llmk_sclkout_3),.rx_sysref(hw.fmc2.llmk_sclkout_3),.txn_out(hw.fmc2.dac_lane_n),.txp_out(hw.fmc2.dac_lane_p),.axi_adc0(axi_fmc2_adc0),.axi_adc1(axi_fmc2_adc1),.axi_dac(axi_fmc2_dac),.rx_reset(rx_reset),.tx_reset(tx_reset),.rx_sys_reset(rx_sys_reset),.tx_sys_reset(tx_sys_reset),.rx_sync(rx_sync_2),.tx_sync(hw.fmc2.dac_sync_req_to_fpga),.adc0(adc4),.adc1(adc5),.adc2(adc6),.adc3(adc7),.dac0(dac4),.dac1(dac5),.dac2(dac6),.dac3(dac7),.rx_aresetn_0(),.rx_aresetn_1(),.rx_frame_error(),.tx_aresetn(),.tx_tready(fmc2_tx_tready),.adc01_valid(),.adc23_valid(),.common0_qpll_lock_out(),.common1_qpll_lock_out(),.reset_done(jesd_reset_done_2),.reset_status(jesd_reset_status_2)
);
wire jesdreset0_done_orig=&{common0_qpll_lock_out_1, common1_qpll_lock_out_2}|master;
wire jesdreset1_done_orig=&{jesd_reset_done_1,jesd_reset_done_2}|master;
reg jesdreset_done0=0;
reg jesdreset_done1=0;
always @(posedge ethclk) begin
	{jesdreset_done0,jesdreset_done1}<={jesdreset0_done_orig,jesdreset1_done_orig};
end
//alatch #(.DWIDTH(2)) jesdresetlatch(.clk(hw.vc707.sysclk),.datain(jesdreset),.dataout({jesdreset0_done,jesdreset1_done}));

flag_xdomain jesdreset0xdomain(.clk2(hw.vc707.sysclk),.clk1(ethclk),.flagin_clk1(jesdreset_done0),.flagout_clk2(jesdreset0_done));
flag_xdomain jesdreset1xdomain(.clk2(hw.vc707.sysclk),.clk1(ethclk),.flagin_clk1(jesdreset_done1),.flagout_clk2(jesdreset1_done));
`else
assign jesdreset0_done=1'b1;
assign jesdreset1_done=1'b1;
`endif
//end
//endgenerate



//*/
reg [31:0] dclkcnt=0;
always @(posedge dspclk) begin
	dclkcnt<=dclkcnt+1;
end
wire sclk=dclkcnt[lbreg.sclkdclkdiv];
assign hw.fmc1.fpga_sync_out_to_trigmux=sclk;//dclkcnt[6];
assign hw.fmc2.fpga_sync_out_to_trigmux=sclk;//dclkcnt[6];
OBUFDS obufds_user_sma_clk(.I(phrefclk125),.O(hw.vc707.user_sma_clock_p),.OB(hw.vc707.user_sma_clock_n));
//OBUFDS obufds_user_sma_gpio(.I(hw.vc707.iic.scl/*sclk*/),.O(hw.vc707.user_sma_gpio_p),.OB(hw.vc707.user_sma_gpio_n));

//assign hw.vc707.rec_clock= &i2cinitdone & lbreg.recclk ? hw.fmc2.lmk_dclk10_m2c_to_fpga : 0;
assign hw.vc707.rec_clock= phrefclk125;//&i2cinitdone & lbreg.recclk ? hw.fmc2.lmk_dclk10_m2c_to_fpga : 0;
//assign hw.vc707.rec_clock= hw.fmc2.lmk_dclk10_m2c_to_fpga;
//OBUFDS obufds_rec_clk(.I(1'b0),.O(hw.vc707.rec_clock_c_p),.OB(hw.vc707.rec_clock_c_n));
wire [63:0] trigcnt;
reg [63:0] fmc2_dclk10cnt=0;
always @(posedge hw.fmc2.lmk_dclk10_m2c_to_fpga) begin
	fmc2_dclk10cnt<=fmc2_dclk10cnt+1;
end
//assign trigcnt = master ? sfpicc.txusrclkcnt : sfpicc.txusrclkcnt_corr;//fmc2_dclk10cnt;
wire locked= (~|sfpicc.cdiff2[47:1] | &sfpicc.cdiff2);  // 0, 1, -1
assign trigcnt = master ? dspclkcnt : dspclkcntcorr;


assign hw.vc707.user_sma_gpio_p=lbreg.scopeselp==16'b0 ? fmc2llmkdclkout2cnt //phrefclk125;//hw.fmc1.lmk_dclk10_m2c_to_fpga;
: lbreg.scopeselp==16'd1 ? phrefclk125
: lbreg.scopeselp==16'd2 ? sfpicc.gticc.txusrclk
: lbreg.scopeselp==16'd3 ? sfpicc.gticc.rxusrclk
: lbreg.scopeselp==16'd4 ? txoutclk_sfp
:0 ;
assign hw.vc707.user_sma_gpio_n=
	lbreg.scopeseln==16'b0 ? (master | (locked & sfpicc.usecorr)) ? ~|trigcnt[9:0] : 0
: lbreg.scopeseln==16'd1 ? phrefclk125
: lbreg.scopeseln==16'd2 ? sfpicc.gticc.txusrclk
: lbreg.scopeseln==16'd3 ? sfpicc.gticc.rxusrclk
: lbreg.scopeseln==16'd4 ? txoutclk_sfp
:0 ;

assign hw.vc707.gpio_led_0=master;
assign master=hw.vc707.gpio_dip_sw0;

assign hw.vc707.si5324_rst=lbreg.si5324_rst;

//assign phrefclk=sgmiiclk;//hw.fmc1.lmk_dclk10_m2c_to_fpga;
//assign phrefclkdiv2=sgmiiclkcnt[0];//fmc1dclk10cnt[0];

localparam NFCNT = 19;
wire [28*NFCNT-1:0] freq_cnt;
assign {lbreg.freq_lb
,lbreg.freq_sgmiiclk
,lbreg.freq_sma_mgt_refclk
,lbreg.freq_si5324_out_c

//,lbreg.freq_si5324_out_div2
//,lbreg.freq_smamgtclk_div2
,lbreg.freq_pcie_clk_qo
,lbreg.freq_user_clock

,lbreg.freq_fmc1_llmk_dclkout_2
,lbreg.freq_fmc1_llmk_sclkout_3
,lbreg.freq_fmc1_lmk_dclk8_m2c_to_fpga
,lbreg.freq_fmc1_lmk_dclk10_m2c_to_fpga

,lbreg.freq_fmc2_llmk_dclkout_2
,lbreg.freq_fmc2_llmk_sclkout_3
,lbreg.freq_fmc2_lmk_dclk8_m2c_to_fpga
,lbreg.freq_fmc2_lmk_dclk10_m2c_to_fpga

,lbreg.freq_rxusrclk_sfp
,lbreg.freq_txusrclk_sfp
,lbreg.freq_rxusrclk_smasfp
,lbreg.freq_txusrclk_smasfp

,lbreg.freq_ethclk
}=freq_cnt;
wire [NFCNT-1:0] freqcnt_clks = {
lbreg.lb.clk
,sgmiiclk
,sma_mgt_refclk
,si5324_out_c

//,si5324_out_div2
//,smamgtclk_div2
,pcie_clk_qo
,user_clock

,dspclk
,hw.fmc1.llmk_sclkout_3
,hw.fmc1.lmk_dclk8_m2c_to_fpga
,hw.fmc1.lmk_dclk10_m2c_to_fpga

,hw.fmc2.llmk_dclkout_2
,hw.fmc2.llmk_sclkout_3
,hw.fmc2.lmk_dclk8_m2c_to_fpga
,hw.fmc2.lmk_dclk10_m2c_to_fpga

,sfpicc.gticc.rxusrclk //_sfp
,sfpicc.gticc.txusrclk //_sfp
,smasfpicc.gticc.rxusrclk //_smasfp
,smasfpicc.gticc.txusrclk //_smasfp

,ethclk
};

genvar jx;
generate for (jx=0; jx<NFCNT; jx=jx+1)
    begin: gen_fcnt
        freq_count freq_count(.clk(freqcnt_clks[jx])
		,.usbclk(lbreg.lb.clk)
		,.frequency(freq_cnt[jx*28+27:jx*28])
		,.diff_stream()
		,.diff_stream_strobe()
		,.glitch_catcher()
        );
    end
endgenerate



/*reg [3*7-1:0] daddrsr={7'h0,7'h14,7'h1c};
localparam RESETCNT=200;
wire reset_in=clk200cnt==RESETCNT;
reg den_in_0=0;//&clk200cnt[9:0];
reg reset_d=0;
wire den_in=den_in_0;//&clk200cnt[9:0];
//wire den_in=&clk200cnt[9:0];
wire dwe_in=1'b0;
wire [6:0] daddr_in;//=7'h0;
wire [15:0] di_in=0;
wire drdy_out;
reg firstreset=1'b1;
reg [31:0] xadcupdatecnt=0;
wire xadcupdatestb=~|xadcupdatecnt;
reg xadcupdate_r=0;
reg xadcupdate_d=0;
wire xadcupdate=~xadcupdate_r&xadcupdate_d;
always @(posedge clk200) begin
	reset_d<=reset_in;
	if (clk200cnt==RESETCNT) begin
		firstreset<=0;
	end
	xadcupdatecnt<=(~|xadcupdatecnt | firstreset) ? (lb.xadcupdate-1) : (xadcupdatecnt-1);
	if (xadcupdatestb) begin
		xadcupdate_r<=1'b1;
	end
	else begin
		if (~drdy_out) begin
			xadcupdate_r<=1'b0;
		end
	end
	xadcupdate_d<=xadcupdate_r;
	if (xadcupdate) begin
		daddrsr<={daddrsr[13:0],daddrsr[20:14]};
	end
	den_in_0<=xadcupdate;
end
assign daddr_in=daddrsr[6:0];
wire [15:0] do_out;
wire [4:0] channel_out;
wire eoc_out;
wire alarm_out;
wire eos_out;
wire busy_out;
xadc_qubic xadc(
 .di_in(di_in),       // input wire [15 : 0] di_in
 .daddr_in(daddr_in),    // input wire [6 : 0] daddr_in
 .den_in(den_in),      // input wire den_in
 .dwe_in(dwe_in),      // input wire dwe_in
 .drdy_out(drdy_out),    // output wire drdy_out
 .do_out(do_out),      // output wire [15 : 0] do_out
 .dclk_in(clk200),     // input wire dclk_in
 .reset_in(reset_in),    // input wire reset_in
 .vp_in(hw.vc707.VP_0),       // input wire vp_in
 .vn_in(hw.vc707.VN_0),       // input wire vn_in
 .channel_out(channel_out), // output wire [4 : 0] channel_out
 .eoc_out(eoc_out),     // output wire eoc_out
 .alarm_out(alarm_out),   // output wire alarm_out
 .eos_out(eos_out),     // output wire eos_out
 .busy_out(busy_out)    // output wire busy_out
);
reg [15:0] xadctemp=0;
reg [15:0] xadcaux4=0;
reg [15:0] xadcaux12=0;
always @(posedge clk200) begin
	if (drdy_out) begin
		case (daddr_in)
			7'h0: xadctemp<=do_out;
			7'h14: xadcaux4<=do_out;
			7'h1c: xadcaux12<=do_out;
		endcase
	end
end

assign lb.xadctemp=xadctemp;
assign lb.xadcaux4=xadcaux4;
assign lb.xadcaux12=xadcaux12;
reg [NSTEP-1:0] dbdone=0;
reg [NSTEP-1:0] dbdonestrobe=0;
reg [NSTEP-1:0] dberror=0;
reg [NSTEP-1:0] dbresetout=0;
reg [NSTEP-1:0] dbdonecriteria=0;
reg [NSTEP-1:0] dbreadycriteria=0;
reg [NSTEP-1:0] dbresetin=0;
always @(posedge hw.vc707.sysclk) begin
	dbdone<=done;
	dbdonestrobe<=donestrobe;
	dberror<=error;
	dbresetout<=resetout;
	dbdonecriteria<=donecriteria;
	dbreadycriteria<=readycriteria;
	dbresetin<=resetin;
end
*/
///*
wire stb_bufreadtestreset;
areset bufreadtestarset(.clk(clk250),.areset(lbreg.stb_bufreadtestreset),.sreset(stb_bufreadtestreset));
wire bufreadtestfull;
bufread #(.AWW(10)
,.DWW(32)
,.DWR(32)) bufreadtest(.wclk(clk250)
,.rclk(udplb.clk)
,.wdata(clk250cnt)
,.waddr(0)
,.wen(1'b1)
,.ren(lbreg.bufreadtest__en)
,.raddr(lbreg.bufreadtest__addr)
,.rdata(lbreg.bufreadtest__data)
,.full(bufreadtestfull)
,.reset(stb_bufreadtestreset));
alatch #(.DWIDTH(1)) bufreadtestfulllatch(.clk(ethclk),.datain(bufreadtestfull),.dataout(lbreg.bufreadtestfull));
wire stb_adc0bufreset;
areset adc0bufareset(.clk(dspclk),.areset(lbreg.stb_adc0bufreset),.sreset(stb_adc0bufreset));
bufread #(.AWW(10)
,.DWW(32)
,.DWR(32)) adc0buf(.wclk(dspclk)
,.rclk(udplb.clk)
,.wdata(adc0[31:0])
,.waddr(0)
,.wen(1'b1)
,.ren(lbreg.adc0buf__en)
,.raddr(lbreg.adc0buf__addr)
,.rdata(lbreg.adc0buf__data)
,.full(lbreg.adc0buffull)
,.reset(stb_adc0bufreset));

assign dsp.clk=dspclk;;
assign dac0=dsp.dac0;
assign dac1=dsp.dac1;
assign dac2=dsp.dac2;
assign dac3=dsp.dac3;
assign dac4=dsp.dac4;
assign dac5=dsp.dac5;
assign dac6=dsp.dac6;
assign dac7=dsp.dac7;
assign dsp.adc0=adc0;
assign dsp.adc1=adc1;
assign dsp.adc2=adc2;
assign dsp.adc3=adc3;
assign dsp.adc4=adc4;
assign dsp.adc5=adc5;
assign dsp.adc6=adc6;
assign dsp.adc7=adc7;


wire [NSTEP-1:0] done;
wire [NSTEP-1:0] dbdone_w;
reg [NSTEP-1:0] dbdone=0;
wire [NSTEP-1:0] donestrobe;
wire [NSTEP-1:0] error;
wire [NSTEP-1:0] resetout;
wire [NSTEP-1:0] dbresetout;
wire [NSTEP-1:0] donecriteria;
reg [NSTEP-1:0] donecriteria_r=0;
wire stbdone;
wire resetin=udphwreset|uarthwreset|poweronreset|(stbdone & (lbreg.loopreset & ~hw.vc707.gpio_dip_sw1) & ((~&done)| (~|eepromrd) | (&eepromrd)));
reg [31:0] resetcnt=0;
always @(posedge hw.vc707.sysclk) begin
	if (resetin) begin
		resetcnt<=resetcnt+1;
	end
	if (stbdone) begin
		dbdone<=dbdone_w;
		donecriteria_r<=donecriteria;
	end
end
wire [NSTEP*16-1:0] readylength;//={NSTEP{16'd20}};
wire [NSTEP*16-1:0] donelength;//={NSTEP{16'd20}};
wire [NSTEP*16-1:0] resetlength;//={NSTEP{16'd20}};
wire [NSTEP*32-1:0] resettimeout;//={NSTEP{32'b0}};
wire [NSTEP*16-1:0] resettodonecheck;//={16'd10,16'd20,16'd1};

wire chainresetreset_x;
areset chainresetareset(.clk(hw.vc707.sysclk),.areset(resetin),.sreset(chainresetreset_x));
wire [NSTEP-1:0] donecriteria_x;
alatch #(.DWIDTH(NSTEP))donecriteriaxdomain(.clk(hw.vc707.sysclk),.datain(donecriteria),.dataout(donecriteria_x));
wire [3:0] dbchainstate,dbchainnext;
chainreset #(.NSTEP(NSTEP))
chainreset(.clk(hw.vc707.sysclk)
,.reset(chainresetreset_x)
,.resetout(resetout)
,.donecriteria(donecriteria_x)
,.resetlength(resetlength)
,.readylength(readylength)
,.donelength(donelength)
,.resettodonecheck(resettodonecheck)
,.resettimeout(resettimeout)
,.done(done)
,.stbdone(stbdone)
,.dbdone(dbdone_w)
,.dbstate(dbchainstate)
,.dbnext(dbchainnext)
,.dbresetout(dbresetout)
);

wire [NSTEP-1:0] done_w;
wire reset_done;

data_xdomain #(.DWIDTH(NSTEP)) donexdomainethclk(.clkin(hw.vc707.sysclk), .gatein(1'b1), .datain(done),.clkout(ethclk),.dataout(done_r3));
assign lbreg.hwresetstatus=done_r3;
assign uartreg.hwresetstatus=done_r3;

assign sysclkmmcm_reset=resetout[SYSCLKMMCM_RESET];
assign donecriteria[SYSCLKMMCM_RESET]=sysclkmmcm_locked;
assign donelength[SYSCLKMMCM_RESET*16+:16]=16'd1;assign readylength[SYSCLKMMCM_RESET*16+:16]=16'd30;assign resetlength[SYSCLKMMCM_RESET*16+:16]=16'd20;assign resettodonecheck[SYSCLKMMCM_RESET*16+:16]=16'd10;assign resettimeout[SYSCLKMMCM_RESET*32+:32]=32'h10000000;
assign idelayctrl_reset=resetout[IDELAYCTRL_RESET];
assign donecriteria[IDELAYCTRL_RESET]=idelayctrl_rdy;
assign donelength[IDELAYCTRL_RESET*16+:16]=16'd1;assign readylength[IDELAYCTRL_RESET*16+:16]=16'd30;assign resetlength[IDELAYCTRL_RESET*16+:16]=16'd20;assign resettodonecheck[IDELAYCTRL_RESET*16+:16]=16'd10;assign resettimeout[IDELAYCTRL_RESET*32+:32]=32'h10000000;
assign sgmiieth_reset=resetout[SGMIIETH_RESET];
assign donecriteria[SGMIIETH_RESET]=sgmiieth_resetdone;
assign donelength[SGMIIETH_RESET*16+:16]=16'd1;assign readylength[SGMIIETH_RESET*16+:16]=16'd30;assign resetlength[SGMIIETH_RESET*16+:16]=16'd20;assign resettodonecheck[SGMIIETH_RESET*16+:16]=16'd10;assign resettimeout[SGMIIETH_RESET*32+:32]=32'h10000000;
assign uartreset=resetout[UARTRESET];
assign donecriteria[UARTRESET]=1'b1;
assign donelength[UARTRESET*16+:16]=16'd1;assign readylength[UARTRESET*16+:16]=16'd30;assign resetlength[UARTRESET*16+:16]=16'd20;assign resettodonecheck[UARTRESET*16+:16]=16'd10;assign resettimeout[UARTRESET*32+:32]=32'h10000000;
assign uartlbreset=resetout[UARTLBRESET];
assign donecriteria[UARTLBRESET]=1'b1;
assign donelength[UARTLBRESET*16+:16]=16'd1;assign readylength[UARTLBRESET*16+:16]=16'd30;assign resetlength[UARTLBRESET*16+:16]=16'd20;assign resettodonecheck[UARTLBRESET*16+:16]=16'd10;assign resettimeout[UARTLBRESET*32+:32]=32'h10000000;
assign i2creset=resetout[I2CRESET];
assign donecriteria[I2CRESET]=i2cresetdone;
assign donelength[I2CRESET*16+:16]=16'd1;assign readylength[I2CRESET*16+:16]=16'd30;assign resetlength[I2CRESET*16+:16]=16'd20;assign resettodonecheck[I2CRESET*16+:16]=16'd10;assign resettimeout[I2CRESET*32+:32]=32'h10000000;
assign i2cinitreset[0]=resetout[I2CINITRESET_0];
assign donecriteria[I2CINITRESET_0]=SIM ? 1'b1 : i2cinitdone[0];
assign donelength[I2CINITRESET_0*16+:16]=16'd1;assign readylength[I2CINITRESET_0*16+:16]=16'd30;assign resetlength[I2CINITRESET_0*16+:16]=16'd20;assign resettodonecheck[I2CINITRESET_0*16+:16]=16'd10;assign resettimeout[I2CINITRESET_0*32+:32]=32'h80000000;
assign phy_reset=resetout[PHYRESET];
assign donecriteria[PHYRESET]=1'b1;
assign donelength[PHYRESET*16+:16]=16'd1;assign readylength[PHYRESET*16+:16]=16'd30;assign resetlength[PHYRESET*16+:16]=16'd20;assign resettodonecheck[PHYRESET*16+:16]=16'd10;assign resettimeout[PHYRESET*32+:32]=32'h800;
assign phy_resetwait=resetout[PHYRESETWAIT5MS];
assign donecriteria[PHYRESETWAIT5MS]=phy_resetwait_done;
assign donelength[PHYRESETWAIT5MS*16+:16]=16'd1;assign readylength[PHYRESETWAIT5MS*16+:16]=16'd30;assign resetlength[PHYRESETWAIT5MS*16+:16]=16'd20;assign resettodonecheck[PHYRESETWAIT5MS*16+:16]=16'd10;assign resettimeout[PHYRESETWAIT5MS*32+:32]=SIM ?  32'h800 : 32'h8000000;
assign mdioreset=resetout[MDIORESET];
assign donecriteria[MDIORESET]=SIM ? 1'b1 : mdioinitdone;
assign donelength[MDIORESET*16+:16]=16'd1;assign readylength[MDIORESET*16+:16]=16'd30;assign resetlength[MDIORESET*16+:16]=16'd20;assign resettodonecheck[MDIORESET*16+:16]=16'd10;assign resettimeout[MDIORESET*32+:32]=32'h80000000;
assign loadmacip=resetout[LOADMACIP];
assign donecriteria[LOADMACIP]=maciploaded;
assign donelength[LOADMACIP*16+:16]=16'd1;assign readylength[LOADMACIP*16+:16]=16'd30;assign resetlength[LOADMACIP*16+:16]=16'd20;assign resettodonecheck[LOADMACIP*16+:16]=16'd10;assign resettimeout[LOADMACIP*32+:32]=32'h10000000;
assign ethreset_w=resetout[ETHRESET_W];
assign donecriteria[ETHRESET_W]=ethresetdone;
assign donelength[ETHRESET_W*16+:16]=16'd1;assign readylength[ETHRESET_W*16+:16]=16'd30;assign resetlength[ETHRESET_W*16+:16]=16'd20;assign resettodonecheck[ETHRESET_W*16+:16]=16'd10;assign resettimeout[ETHRESET_W*32+:32]=32'h10000000;
assign i2cinitreset[1]=resetout[I2CINITRESET_1];
assign donecriteria[I2CINITRESET_1]=SIM ? 1'b1 : i2cinitdone[1];
assign donelength[I2CINITRESET_1*16+:16]=16'd1;assign readylength[I2CINITRESET_1*16+:16]=16'd30;assign resetlength[I2CINITRESET_1*16+:16]=16'd20;assign resettodonecheck[I2CINITRESET_1*16+:16]=16'd10;assign resettimeout[I2CINITRESET_1*32+:32]=32'h80000000;
assign axireset_sys=resetout[AXIRESET];
assign donecriteria[AXIRESET]=1'b1;
assign donelength[AXIRESET*16+:16]=16'd1;assign readylength[AXIRESET*16+:16]=16'd30;assign resetlength[AXIRESET*16+:16]=16'd20;assign resettodonecheck[AXIRESET*16+:16]=16'd10;assign resettimeout[AXIRESET*32+:32]=32'h10000000;
assign jesdreset0=resetout[JESDRESET0];
assign donecriteria[JESDRESET0]=jesdreset0_done;
assign donelength[JESDRESET0*16+:16]=16'd1;assign readylength[JESDRESET0*16+:16]=16'd30;assign resetlength[JESDRESET0*16+:16]=16'd20;assign resettodonecheck[JESDRESET0*16+:16]=16'd10;assign resettimeout[JESDRESET0*32+:32]=32'h10000000;
assign jesdreset1=resetout[JESDRESET1];
assign donecriteria[JESDRESET1]=jesdreset1_done;
assign donelength[JESDRESET1*16+:16]=16'd1;assign readylength[JESDRESET1*16+:16]=16'd30;assign resetlength[JESDRESET1*16+:16]=16'd20;assign resettodonecheck[JESDRESET1*16+:16]=16'd10;assign resettimeout[JESDRESET1*32+:32]=32'h10000000;
assign axiinitreset=resetout[AXIINITRESET];
assign donecriteria[AXIINITRESET]=axiinitdone|master;
assign donelength[AXIINITRESET*16+:16]=16'd1;assign readylength[AXIINITRESET*16+:16]=16'd30;assign resetlength[AXIINITRESET*16+:16]=16'd20;assign resettodonecheck[AXIINITRESET*16+:16]=16'd10;assign resettimeout[AXIINITRESET*32+:32]=32'h10000000;
assign i2cinitreset[2]=resetout[I2CINITRESET_2];
assign donecriteria[I2CINITRESET_2]=SIM ? 1'b1 : i2cinitdone[2];
assign donelength[I2CINITRESET_2*16+:16]=16'd1;assign readylength[I2CINITRESET_2*16+:16]=16'd30;assign resetlength[I2CINITRESET_2*16+:16]=16'd20;assign resettodonecheck[I2CINITRESET_2*16+:16]=16'd10;assign resettimeout[I2CINITRESET_2*32+:32]=32'h80000000;
//assign qpllreset_113=resetout[QPLLRESET_113];
//assign donecriteria[QPLLRESET_113]=qpllresetdone_113;
//assign donelength[QPLLRESET_113*16+:16]=16'd30;
//assign readylength[QPLLRESET_113*16+:16]=16'd30;assign resetlength[QPLLRESET_113*16+:16]=16'd20;assign resettodonecheck[QPLLRESET_113*16+:16]=16'd10;assign resettimeout[QPLLRESET_113*32+:32]=32'h10000000;
assign reset_sfp=resetout[RESET_SFP];
assign donecriteria[RESET_SFP]=sfpicc.resetdone;//_sfp;
assign donelength[RESET_SFP*16+:16]=16'd1;assign readylength[RESET_SFP*16+:16]=16'd30;assign resetlength[RESET_SFP*16+:16]=16'd20;assign resettodonecheck[RESET_SFP*16+:16]=16'd10;assign resettimeout[RESET_SFP*32+:32]=32'h10000000;
assign reset_done=resetout[RESETDONE];
assign donecriteria[RESETDONE]=1'b1;//_sfp;
assign donelength[RESETDONE*16+:16]=16'd1;assign readylength[RESETDONE*16+:16]=16'd30;assign resetlength[RESETDONE*16+:16]=16'd20;assign resettodonecheck[RESETDONE*16+:16]=16'd10;assign resettimeout[RESETDONE*32+:32]=32'h10000000;
assign reset_smasfp=resetout[RESET_SMASFP];
assign donecriteria[RESET_SMASFP]=smasfpicc.resetdone;//resetdone_smasfp;
assign donelength[RESET_SMASFP*16+:16]=16'd1;assign readylength[RESET_SMASFP*16+:16]=16'd30;assign resetlength[RESET_SMASFP*16+:16]=16'd20;assign resettodonecheck[RESET_SMASFP*16+:16]=16'd10;assign resettimeout[RESET_SMASFP*32+:32]=32'h10000000;

assign i2cinitreset[3]=resetout[I2CINITRESET_3];
assign donecriteria[I2CINITRESET_3]=SIM ? 1'b1 : i2cinitdone[3];
assign donelength[I2CINITRESET_3*16+:16]=16'd1;assign readylength[I2CINITRESET_3*16+:16]=16'd30;assign resetlength[I2CINITRESET_3*16+:16]=16'd20;assign resettodonecheck[I2CINITRESET_3*16+:16]=16'd10;assign resettimeout[I2CINITRESET_3*32+:32]=32'h80000000;
//assign qpllreset_114=resetout[QPLLRESET_114];
//assign donecriteria[QPLLRESET_114]=qpllresetdone_114;
//assign donelength[QPLLRESET_114*16+:16]=16'd30;
//assign readylength[QPLLRESET_114*16+:16]=16'd30;assign resetlength[QPLLRESET_114*16+:16]=16'd20;assign resettodonecheck[QPLLRESET_114*16+:16]=16'd10;assign resettimeout[QPLLRESET_114*32+:32]=32'h10000000;
//assign pll_reset=resetout[HELPPLL];
//assign donecriteria[HELPPLL]=pll_locked;
//assign donelength[HELPPLL*16+:16]=16'd30;
//assign readylength[HELPPLL*16+:16]=16'd30;assign resetlength[HELPPLL*16+:16]=16'd20;assign resettodonecheck[HELPPLL*16+:16]=16'd10;assign resettimeout[HELPPLL*32+:32]=32'h10000000;

gitrevision gitrevision(lbreg.gitrevision);

//`include "ilaadcauto.vh"
//`include "ilaauto.vh"
//`include "ilahelpauto.vh"
//`include "ilaethauto.vh"
//`include "ilasysauto.vh"
//`include "ilaiccauto.vh"
endmodule
