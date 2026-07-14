module qubicdsp #(parameter DEBUG="false"
)(dsp.dsp dsp
,regmap.dsp lbreg
);


wire lb_clk;
assign lb_clk=lbreg.lb.clk;
wire [19:0] lb_addr;
assign lb_addr=lbreg.lb.waddr;
wire lb_write;
assign lb_write=lbreg.lb.write;
wire lb_read;
assign lb_read=lbreg.lb.read;
wire [31:0] lb_wdata;
assign lb_wdata=lbreg.lb.wdata;

reg [31:0] trig_cnts;
wire trig_chan;
assign trig_chan = trig_cnts == lbreg.period_dac0-1;
initial begin
	trig_cnts = 32'd0;
end
always @(posedge dsp.clk) begin
	if (lbreg.stb_dsp_reset) begin
		trig_cnts <= 32'd0;
	end else begin
		trig_cnts <= trig_chan ? 0 : trig_cnts + 1'b1;
    end
end

// Move localbus writes to dsp_clk domain for phalanx's use
// if aw==10, addr base of 0x38000 yields lb_addr[19:15] == 7
wire phalanx_gstrobe0 = lb_write;// & (lb_addr[19:10+5] == 7);
wire phalanx_gstrobe;
wire [10+9:0] d_addr;
wire [31:0] d_wdata;
wire [0:0] d_write;
data_xdomain #(.size(20+32)) dxw(.clk_in(lb_clk),
	.gate_in(phalanx_gstrobe0),
	.data_in({lb_addr[19:0], lb_wdata}),
	.clk_out(dsp.clk), .gate_out(phalanx_gstrobe),
	.data_out({d_addr, d_wdata})
);
assign d_write=phalanx_gstrobe;

// Command memory - temporary placeholder for a real program generator
parameter CMDAW=18;
//wire phalanx_cstrobe = d_write & (d_addr[19:CMDAW]==2'b10);// 0x80000 to 0xbffff
wire phalanx_cstrobe = d_write & (d_addr[19:CMDAW]==2'b01);// 0x40000 to 0x7ffff
wire cstrobe;
wire [7:0] cmda;
wire [63:0] command;
wire  [31:0]  extra;
reg cstrobe_d=0;
reg [7:0] cmda_d=0;
reg [63:0] command_d=0;
reg  [31:0]  extra_d=0;
wire cstrobe_w;
wire [7:0] cmda_w;
wire [63:0] command_w;
wire  [31:0]  extra_w;

qcmd_gen #(.aw(CMDAW))
qcmd(.clk(dsp.clk),
	.waddr(d_addr[CMDAW-1:0]), .wdata(d_wdata), .wstrobe(phalanx_cstrobe), .trig(trig_chan),
	//.command(command), .cmda(cmda), .cstrobe(cstrobe), .extra(extra)
	.command(command_w), .cmda(cmda_w), .cstrobe(cstrobe_w), .extra(extra_w)
);
always @(posedge dsp.clk) begin
	command_d <= command_w;
	cmda_d <= cmda_w;
	cstrobe_d <= cstrobe_w;
	extra_d <= extra_w;
end
assign cstrobe=cstrobe_d;
assign cmda=cmda_d;
assign command=command_d;
assign extra=extra_d;

wire [3:0]resultx;
wire [3:0]resulty;
wire daczero=extra[0]&resultx[0];
localparam dw=16;
localparam nel=8;  // maybe move to 16 later
localparam nell=3;  // maybe move to 16 later
localparam qbits=4;
localparam tslice=4;
wire fault0;
wire [qbits*2*tslice*dw-1:0] dacout;
//wire phalanx_wstrobe = d_write & (d_addr[19:15]==6'b00111);  // 0x38000
wire phalanx_wstrobe = d_write & ((d_addr[19:12]==8'h06)|(d_addr[19:12]==8'h07)|(d_addr[19:12]==8'h08)|(d_addr[19:12]==8'h09)|(d_addr[19:12]==8'h0a)|(d_addr[19:12]==8'h0b)|(d_addr[19:12]==8'h0c)|(d_addr[19:12]==8'h0d));  // 0x6000,0x7000,0x8000,0x9000,0xa000,0xb000,0xc000,0xd000
phalanx #(.aw(10), .dw(dw), .nel(nel), .qbits(qbits), .tslice(tslice)) phalanx(
	.clk(dsp.clk), .fault(fault0),
	.command(command), .cmda(cmda[nell-1:0]), .cstrobe(cstrobe&~cmda[nell]),
	.waddr(d_addr[10+3+2-1:0]), .wdata(d_wdata), .wstrobe(phalanx_wstrobe),
	.dacout(dacout), .daczero(daczero)
);

wire [qbits*2*tslice*dw-1:0] dacout_d;
wire [16*8-1:0] dac_dc;
assign dac_dc={lbreg.dac7_dc,lbreg.dac6_dc,lbreg.dac5_dc,lbreg.dac4_dc,lbreg.dac3_dc,lbreg.dac2_dc,lbreg.dac1_dc,lbreg.dac0_dc};
genvar idac;
genvar islice;
generate
for (idac=0; idac<8; idac=idac+1) begin: gen_dac
	for (islice=0; islice<4; islice=islice+1) begin: gen_slice
		wire signed [15:0] dacout_16=dacout[(idac)*64+(islice+1)*16-1:(idac*64)+islice*16];
		wire signed [15:0] dacdc_16 = dac_dc[(idac+1)*16-1:idac*16];
		reg signed [15:0] dacout_d16=0;
		always @(posedge dsp.clk) begin
			dacout_d16 <= dacout_16+dacdc_16;
		end
		assign dacout_d[(idac)*64+(islice+1)*16-1:(idac*64)+islice*16] = dacout_d16;
	end
end
endgenerate
localparam NMEAS=4;
wire [NMEAS-1:0] meas_wstrobe;
//assign meas_wstrobe[0] = d_write & (d_addr[19:12]==8'hc0);  //  0xc0000
//assign meas_wstrobe[1] = d_write & (d_addr[19:12]==8'hc4);  //  0xc4000
//assign meas_wstrobe[2] = d_write & (d_addr[19:12]==8'hc8);  //  0xc8000
//assign meas_wstrobe[3] = d_write & (d_addr[19:12]==8'hcc);  //  0xcc000
assign meas_wstrobe[0] = d_write & (d_addr[19:12]==8'h10);  //  0x10000
assign meas_wstrobe[1] = d_write & (d_addr[19:12]==8'h14);  //  0x14000
assign meas_wstrobe[2] = d_write & (d_addr[19:12]==8'h18);  //  0x18000
assign meas_wstrobe[3] = d_write & (d_addr[19:12]==8'h1c);  //  0x1c000
wire [NMEAS-1:0] meas_cstrobe ;
assign meas_cstrobe[0]= cstrobe & (cmda==8'h8);
assign meas_cstrobe[1]= cstrobe & (cmda==8'h9);
assign meas_cstrobe[2]= cstrobe & (cmda==8'ha);
assign meas_cstrobe[3]= cstrobe & (cmda==8'hb);
wire [16*tslice-1:0] xmeasin;
wire [16*tslice-1:0] ymeasin;

assign xmeasin=lbreg.digiloopback ? dsp.dac0 : dsp.adc0;
assign ymeasin=lbreg.digiloopback ? dsp.dac1 : dsp.adc1;

wire [16*tslice-1:0] xmeasin_d, ymeasin_d;
reg_delay #(.DW(16*tslice),.LEN(5)) delay_xmeasin(.clk(dsp.clk),.din(xmeasin),.dout(xmeasin_d),.gate(1'b1));
reg_delay #(.DW(16*tslice),.LEN(5)) delay_ymeasin(.clk(dsp.clk),.din(ymeasin),.dout(ymeasin_d),.gate(1'b1));

wire [dw-1:0] adc0_min, adc1_min;
wire [dw-1:0] adc0_max, adc1_max;
minmax2 #(.dw(dw),.n(tslice)) adc0_minmax(.clk(dsp.clk), .xin(dsp.adc0), .reset(meas_cstrobe[0]), .xmin(adc0_min), .xmax(adc0_max));
minmax2 #(.dw(dw),.n(tslice)) adc1_minmax(.clk(dsp.clk), .xin(dsp.adc1), .reset(meas_cstrobe[0]), .xmin(adc1_min), .xmax(adc1_max));
assign lbreg.adc0_min=adc0_min;
assign lbreg.adc0_max=adc0_max;
assign lbreg.adc1_min=adc1_min;
assign lbreg.adc1_max=adc1_max;

wire [16*tslice-1:0] xbase [3:0];
wire [16*tslice-1:0] ybase [3:0];
wire [16*tslice-1:0] xlo_w [3:0];
wire [16*tslice-1:0] ylo_w [3:0];
reg [16*tslice-1:0] xlo [3:0];
reg [16*tslice-1:0] ylo [3:0];
wire [16+16-1:0] xacc [3:0];
wire [16+16-1:0] yacc [3:0];
wire [NMEAS-1:0] meas_active_w;
wire [NMEAS-1:0] meas_done;
genvar imeas;
generate
for (imeas=0; imeas<NMEAS; imeas=imeas+1) begin: gen_meas
	meas2 #(.tslice(tslice),.qbits(qbits),.aw(12),.dw(dw))
	meas2(.clk(dsp.clk)
,.active(meas_active_w[imeas])
,.collision()
,.command(command)
,.cstrobe(meas_cstrobe[imeas])
,.reset(meas_cstrobe[imeas])
,.waddr(d_addr[13:0])
,.wdata(d_wdata)
,.wstrobe(meas_wstrobe[imeas])
,.xacc(xacc[imeas])
,.yacc(yacc[imeas])
,.xmeasin(xmeasin_d)
,.ymeasin(ymeasin_d)
,.xbase(xbase[imeas])
,.ybase(ybase[imeas])
,.xlo(xlo_w[imeas])
,.ylo(ylo_w[imeas])
,.xoffset(lbreg.xoffset)
,.yoffset(lbreg.yoffset)
,.iqrot(lbreg.iqrot)
,.done(meas_done[imeas])
,.resultx(resultx[imeas])
,.resulty(resulty[imeas])
);
end
endgenerate
reg [NMEAS-1:0] meas_active_d=0;
wire [NMEAS-1:0] meas_active;
reg_delay #(.DW(NMEAS),.LEN(5)) delay_meas_active(.clk(dsp.clk),.din(meas_active_w),.dout(meas_active),.gate(1'b1));
reg [NMEAS-1:0] meas_done_d=0;
reg [31:0] xacc_d [3:0];
reg [31:0] yacc_d [3:0];
reg [12:0] accaddr [NMEAS-1:0];
wire [NMEAS-1:0] full;
assign full[0]=accaddr[0][12];
assign full[1]=accaddr[1][12];
assign full[2]=accaddr[2][12];
assign full[3]=accaddr[3][12];
assign lbreg.full=full;
reg [NMEAS-1:0] firstrun=0;
wire [NMEAS-1:0] meas_done_2cycle;
assign meas_done_2cycle=meas_done|meas_done_d;
wire [NMEAS-1:0] measxypush;
assign measxypush=~full&meas_done_2cycle;
initial begin
	accaddr[0] = {1'b1,12'b0};
	accaddr[1] = {1'b1,12'b0};
	accaddr[2] = {1'b1,12'b0};
	accaddr[3] = {1'b1,12'b0};
end

wire [31:0] accout [3:0];
genvar iacc;
generate for (iacc=0; iacc<NMEAS; iacc=iacc+1) begin: gen_accbuf
	always @(posedge dsp.clk) begin
		xlo[iacc] <= xlo_w[iacc];
		ylo[iacc] <= ylo_w[iacc];
		xacc_d[iacc] <= xacc[iacc];
		yacc_d[iacc] <= yacc[iacc];
		meas_active_d[iacc] <= meas_active[iacc];
		meas_done_d[iacc] <= meas_done[iacc];
		firstrun[iacc] <= lbreg.stb_start ? 1'b1 : trig_chan ? 1'b0 : firstrun[iacc];
		if (firstrun[iacc])
			accaddr[iacc] <= 0;
		else if (~full[iacc]&meas_done_2cycle[iacc])
			accaddr[iacc] <= accaddr[iacc]+1'b1;
	end
	dpram #(.DW(32),.AW(12),.BUFIN(0),.BUFOUT(1),.SIM(0)) accbuf(.clka(dsp.clk)
	,.addra(accaddr[iacc][11:0])
	,.dina(accaddr[iacc][0] ? yacc_d[iacc] : xacc[iacc])
	,.wena(measxypush[iacc])
	,.clkb(lb_clk)
	,.addrb(lb_addr[11:0])
	,.doutb(accout[iacc])
	,.douta()
	,.renb(1'b1)
	,.reset()
	);
end
endgenerate

assign lbreg.accout_0__data=accout[0];
assign lbreg.accout_1__data=accout[1];
assign lbreg.accout_2__data=accout[2];
assign lbreg.accout_3__data=accout[3];

localparam NMON=2;
localparam MEMAW=10;
wire mon_ena;
wire [MEMAW-1:0] mon_addr;
reg [dw-1:0] mon_in1=0,mon_in0=0;
wire [dw*NMON-1:0] mon_in,mon_out;
wire [dw-1:0] mon0_1slice [15:0];
wire [dw-1:0] mon1_1slice [15:0];
reg [dw*tslice-1:0] mon_4slice [31:0];
integer inittrigcnts=0;
initial begin
    for (inittrigcnts=0; inittrigcnts<32; inittrigcnts=inittrigcnts+1)
        mon_4slice[inittrigcnts] = {dw*tslice{1'b0}};
end
always @(posedge dsp.clk) begin
	mon_4slice[0] <= xmeasin_d;
	mon_4slice[1] <= ymeasin_d;
	mon_4slice[2] <= xlo_w[0];
	mon_4slice[3] <= ylo_w[0];
	mon_4slice[4] <= xlo_w[1];
	mon_4slice[5] <= ylo_w[1];
	mon_4slice[6] <= xlo_w[2];
	mon_4slice[7] <= ylo_w[2];
	mon_4slice[8] <= dsp.dac0;
	mon_4slice[9] <= dsp.dac1;
	mon_4slice[10] <= dsp.dac2;
	mon_4slice[11] <= dsp.dac3;
	mon_4slice[12] <= dsp.dac4;
	mon_4slice[13] <= dsp.dac5;
	mon_4slice[14] <= dsp.dac6;
	mon_4slice[15] <= dsp.dac7;
end
genvar imux;
generate for (imux=0; imux<32; imux=imux+1) begin: gen_slice_mux
	assign mon0_1slice[imux] = (lbreg.mon_slice==0) ? mon_4slice[imux][15:0] : (lbreg.mon_slice==1) ? mon_4slice[imux][31:16] : (lbreg.mon_slice==2) ? mon_4slice[imux][47:32] : mon_4slice[imux][63:48];
	assign mon1_1slice[imux] = (lbreg.mon_slice==0) ? mon_4slice[imux][15:0] : (lbreg.mon_slice==1) ? mon_4slice[imux][31:16] : (lbreg.mon_slice==2) ? mon_4slice[imux][47:32] : mon_4slice[imux][63:48];
end
endgenerate
reg [15:0] panzoom_testcnt=0;
always @(posedge dsp.clk) begin
	if (trig_chan==1)
		panzoom_testcnt <= 0;
	else
		panzoom_testcnt<=panzoom_testcnt+1;
	case(lbreg.mon_sel0)
		5'h0: mon_in0 <= mon0_1slice[0];
		5'h1: mon_in0 <= mon0_1slice[1];
		5'h2: mon_in0 <= mon0_1slice[2];
		5'h3: mon_in0 <= mon0_1slice[3];
		5'h4: mon_in0 <= mon0_1slice[4];
		5'h5: mon_in0 <= mon0_1slice[5];
		5'h6: mon_in0 <= mon0_1slice[6];
		5'h7: mon_in0 <= mon0_1slice[7];
		5'h8: mon_in0 <= mon0_1slice[8];
		5'h9: mon_in0 <= mon0_1slice[9];
		5'ha: mon_in0 <= mon0_1slice[10];
		5'hb: mon_in0 <= mon0_1slice[11];
		5'hc: mon_in0 <= mon0_1slice[12];
		5'hd: mon_in0 <= mon0_1slice[13];
		5'he: mon_in0 <= mon0_1slice[14];
		5'hf: mon_in0 <= mon0_1slice[15];
		5'h10: mon_in0 <= 16'hdead;
		5'h11: mon_in0 <= panzoom_testcnt;
		default: mon_in0 <= mon0_1slice[0];
	endcase
end
always @(posedge dsp.clk) begin
	case(lbreg.mon_sel1)
		5'h0: mon_in1 <= mon1_1slice[0];
		5'h1: mon_in1 <= mon1_1slice[1];
		5'h2: mon_in1 <= mon1_1slice[2];
		5'h3: mon_in1 <= mon1_1slice[3];
		5'h4: mon_in1 <= mon1_1slice[4];
		5'h5: mon_in1 <= mon1_1slice[5];
		5'h6: mon_in1 <= mon1_1slice[6];
		5'h7: mon_in1 <= mon1_1slice[7];
		5'h8: mon_in1 <= mon1_1slice[8];
		5'h9: mon_in1 <= mon1_1slice[9];
		5'ha: mon_in1 <= mon1_1slice[10];
		5'hb: mon_in1 <= mon1_1slice[11];
		5'hc: mon_in1 <= mon1_1slice[12];
		5'hd: mon_in1 <= mon1_1slice[13];
		5'he: mon_in1 <= mon1_1slice[14];
		5'hf: mon_in1 <= mon1_1slice[15];
		5'h10: mon_in1 <= 16'hbeef;
		5'h11: mon_in1 <= panzoom_testcnt;
		default: mon_in1 <= mon1_1slice[2];
	endcase
end
assign mon_in={mon_in1,mon_in0};
wire stopped;
assign lbreg.stopped=stopped;
panzoom #(.DW(dw),.MAXDAVR(20),.NCHAN(NMON),.MEMAW(MEMAW)) panzoom(.clk(dsp.clk)
,.trig(trig_chan)
,.reset(lbreg.stb_panzoom_reset)
,.din(mon_in)
,.opsel(lbreg.opsel)
,.dout(mon_out)
,.navr(lbreg.mon_navr)
,.dt(lbreg.mon_dt)
,.gout(mon_ena)
,.addrcnt(mon_addr)
,.stopped(stopped)
);
wire [dw-1:0] buf_monout [NMON-1:0];
genvar imon;
generate for (imon=0; imon<NMON; imon=imon+1) begin: gen_monitor
	dpram #(.DW(dw),.AW(MEMAW),.BUFIN(0),.BUFOUT(1),.SIM(0)) monitor(.clka(dsp.clk)
	,.addra(mon_addr)
	,.dina(lbreg.panzoom_test ? panzoom_testcnt : mon_out[(imon+1)*dw-1:imon*dw])
	,.wena(mon_ena)
	,.clkb(lb_clk)
	,.addrb(lb_addr[MEMAW-1:0])
	,.doutb(buf_monout[imon])
	,.douta()
	,.renb(1'b1)
	,.reset()
	);
end
endgenerate

assign lbreg.buf_monout_0__data=buf_monout[0];
assign lbreg.buf_monout_1__data=buf_monout[1];

// Drive the DAC datapath
assign dsp.dac0=dacout_d[0*64 +: 64];  // qubit 0 Q
assign dsp.dac1=dacout_d[1*64 +: 64];  // qubit 0 Q
assign dsp.dac2=dacout_d[2*64 +: 64];  // qubit 1 I
assign dsp.dac3=dacout_d[3*64 +: 64];  // qubit 1 Q
assign dsp.dac4=dacout_d[4*64 +: 64];  // qubit 2 I
assign dsp.dac5=dacout_d[5*64 +: 64];  // qubit 2 Q
assign dsp.dac6=dacout_d[6*64 +: 64];  // qubit 3 I
assign dsp.dac7=dacout_d[7*64 +: 64];  // qubit 3 I


wire [3:0] markpin;
assign {dsp.user_sma_gpio_n,dsp.user_sma_gpio_p}=markpin[1:0];
genvar idigi;
generate for (idigi=0; idigi<4; idigi=idigi+1) begin: digigen
	wire [1:0] idigi_w=idigi;
	wire [7:0] stbdigi={6'b000011,idigi_w};
	wire digi_cstrobe= cstrobe & (cmda==stbdigi);
	digimark mark1(.clk(dsp.clk),.cstrobe(digi_cstrobe),.command(command),.mark(markpin[idigi]));
end
endgenerate

endmodule

interface dsp#(parameter DEBUG="false")();
wire clk;
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
wire user_sma_gpio_p;
wire user_sma_gpio_n;

modport dsp(input clk,adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7
,output dac0,dac1,dac2,dac3,dac4,dac5,dac6,dac7,user_sma_gpio_p,user_sma_gpio_n
);
modport cfg(output clk,adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7
,input dac0,dac1,dac2,dac3,dac4,dac5,dac6,dac7,user_sma_gpio_p,user_sma_gpio_n
);
endinterface

