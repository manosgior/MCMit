//`include "constants.vams"
//`include "xc7vx485tffg1761pkg.vh"
`timescale 1ns / 100ps
module qubic_tb();
localparam SIM=1;
xc7vx485tffg1761pkg fpga();
reg sysclk=0;
integer cc=0;
initial begin
	$dumpfile("qubic.vcd");
	$dumpvars(17,qubic_tb);
//	for (cc=0; cc<300; cc=cc+1) begin
	while (1) begin
		cc=cc+1;

		sysclk=0; #2.5;
		sysclk=1; #2.5;
	end
	$finish();
end

reg sgmiiclk=0;
initial begin
    forever #(4) sgmiiclk=~sgmiiclk;
end
wire si5324clk;
reg clk100=0;
initial begin
    forever #(5) clk100=~clk100;
end
reg clk125=0;
initial begin
    forever #(4) clk125=~clk125;
end
reg ghzclk=0;
initial begin
    forever #(0.5) ghzclk=~ghzclk;
end
localparam DELAY=0.22;
reg clk125_delay=0;
always @(*)
	clk125_delay= #DELAY clk125;
reg clk250=0;
initial begin
    forever #(2) clk250=~clk250;
end
reg [31:0] clk250cnt=0;
always @(posedge clk250) begin
	clk250cnt<=clk250cnt+1;
end

reg helpclk=0;
initial begin
    forever #(8.110488) helpclk=~helpclk;
end
reg clk500=0;
initial begin
    forever #(1) clk500=~clk500;
end
reg [31:0] clk500cnt=0;
always @(posedge clk500) begin
	clk500cnt<=clk500cnt+1;
end
hw  hw();
vc707_sim vc707_sim(.fpga(fpga),.hw(hw.vc707.sim));
fmc120_sim fmc1(.fmcpin(hw.vc707.fmc1pin),.fmc120(hw.fmc1.sim));
fmc120_sim_2 fmc2(.fmcpin(hw.vc707.fmc2pin),.fmc120(hw.fmc2.sim));
wire [1:0] adc0=2'b01;
wire [1:0] adc1=2'b10;
wire [1:0] adc2=2'b01;
wire [1:0] adc3=2'b10;
wire [1:0] adc4=2'b01;
wire [1:0] adc5=2'b10;
wire [1:0] adc6=2'b01;
wire [1:0] adc7=2'b10;

assign {hw.fmc1.sim.adc0_da1_p,hw.fmc1.sim.adc0_da1_n}={hw.fmc1.dac_lane_p[0],hw.fmc1.dac_lane_n[0]};
assign {hw.fmc1.sim.adc0_da2_p,hw.fmc1.sim.adc0_da2_n}={hw.fmc1.dac_lane_p[1],hw.fmc1.dac_lane_n[1]};
assign {hw.fmc1.sim.adc0_db1_p,hw.fmc1.sim.adc0_db1_n}={hw.fmc1.dac_lane_p[2],hw.fmc1.dac_lane_n[2]};
assign {hw.fmc1.sim.adc0_db2_p,hw.fmc1.sim.adc0_db2_n}={hw.fmc1.dac_lane_p[3],hw.fmc1.dac_lane_n[3]};
assign {hw.fmc1.sim.adc1_da1_p,hw.fmc1.sim.adc1_da1_n}={hw.fmc1.dac_lane_p[4],hw.fmc1.dac_lane_n[4]};
assign {hw.fmc1.sim.adc1_da2_p,hw.fmc1.sim.adc1_da2_n}={hw.fmc1.dac_lane_p[5],hw.fmc1.dac_lane_n[5]};
assign {hw.fmc1.sim.adc1_db1_p,hw.fmc1.sim.adc1_db1_n}={hw.fmc1.dac_lane_p[6],hw.fmc1.dac_lane_n[6]};
assign {hw.fmc1.sim.adc1_db2_p,hw.fmc1.sim.adc1_db2_n}={hw.fmc1.dac_lane_p[7],hw.fmc1.dac_lane_n[7]};
wire master=1'b1;
assign {hw.vc707.sma_mgt_refclk_p, hw.vc707.sma_mgt_refclk_n}=master ? {clk250cnt[0],~clk250cnt[0]} : {hw.vc707.user_sma_clock_p, hw.vc707.user_sma_clock_n};
assign {hw.vc707.user_clock_p,hw.vc707.user_clock_n}={helpclk,~helpclk};
assign hw.vc707.gpio_dip_sw0=master;
//assign {hw.fmc2.sim.adc0_da1_p,hw.fmc2.sim.adc0_da1_n}=hw.fmc2.adca_sync_in_l_vadj ? {adc0[0],~adc0[0]} : {hw.fmc2.dac_lane_p[0],hw.fmc2.dac_lane_n[0]};
//assign {hw.fmc2.sim.adc0_da2_p,hw.fmc2.sim.adc0_da2_n}=hw.fmc2.adca_sync_in_l_vadj ? {adc0[1],~adc0[1]} : {hw.fmc2.dac_lane_p[1],hw.fmc2.dac_lane_n[1]};
//assign {hw.fmc2.sim.adc0_db1_p,hw.fmc2.sim.adc0_db1_n}=hw.fmc2.adca_sync_in_l_vadj ? {adc1[0],~adc1[0]} : {hw.fmc2.dac_lane_p[2],hw.fmc2.dac_lane_n[2]};
//assign {hw.fmc2.sim.adc0_db2_p,hw.fmc2.sim.adc0_db2_n}=hw.fmc2.adca_sync_in_l_vadj ? {adc1[1],~adc1[1]} : {hw.fmc2.dac_lane_p[3],hw.fmc2.dac_lane_n[3]};
//assign {hw.fmc2.sim.adc1_da1_p,hw.fmc2.sim.adc1_da1_n}=hw.fmc2.adca_sync_in_l_vadj ? {adc2[0],~adc2[0]} : {hw.fmc2.dac_lane_p[4],hw.fmc2.dac_lane_n[4]};
//assign {hw.fmc2.sim.adc1_da2_p,hw.fmc2.sim.adc1_da2_n}=hw.fmc2.adca_sync_in_l_vadj ? {adc2[1],~adc2[1]} : {hw.fmc2.dac_lane_p[5],hw.fmc2.dac_lane_n[5]};
//assign {hw.fmc2.sim.adc1_db1_p,hw.fmc2.sim.adc1_db1_n}=hw.fmc2.adca_sync_in_l_vadj ? {adc3[0],~adc3[0]} : {hw.fmc2.dac_lane_p[6],hw.fmc2.dac_lane_n[6]};
//assign {hw.fmc2.sim.adc1_db2_p,hw.fmc2.sim.adc1_db2_n}=hw.fmc2.adca_sync_in_l_vadj ? {adc3[1],~adc3[1]} : {hw.fmc2.dac_lane_p[7],hw.fmc2.dac_lane_n[7]};

assign hw.fmc1.dac_sync_req_to_fpga=hw.fmc1.adca_sync_in_l_vadj;//,hw.fmc1.adcb_sync_in_l_vadj
assign hw.fmc2.dac_sync_req_to_fpga=hw.fmc2.adca_sync_in_l_vadj;//,hw.fmc2.adcb_sync_in_l_vadj


wire slaveack1valid;
wire [7:0] slaveack1;
reg [23:0] slavereg=24'hadbeef;
reg slavetx1rx0=0;
wire slaverxvalid;
wire [31:0] slavedatarx;
wire [6:0] i2cslavedevaddr;
wire [7:0] i2cslaveaddr;
wire i2cslavew0r1;
reg [15:0] i2cslavenack_r=0;
assign hw.vc707.iic.sda=qubic.qubichw_config.sdaasrx ? 1'b1 : 1'bz;
/*i2cslave #(.NACK(2)) i2cslave(.clk(sgmiiclk),.SCL(hw.vc707.iic.scl),.SDA(hw.vc707.iic.sda),.rst(1'b0),.datatx({8'h0,slavereg}),.ack1(slaveack1),.ack1valid(slaveack1valid),.datarx(slavedatarx),.rxvalid(slaverxvalid)
,.devaddr(i2cslavedevaddr)
,.addr(i2cslaveaddr)
,.w0r1(i2cslavew0r1)
,.nack(i2cslavenack_r)
);*/
always @(posedge sgmiiclk) begin
	if (slaveack1valid)
		slavetx1rx0<=slaveack1[0];
	if (slaveack1[0]==0)
		if (slaverxvalid)
			slavereg<=slavedatarx[23:0];
	case (i2cslavedevaddr)
		7'h74: i2cslavenack_r<=16'h2;
		7'h54: i2cslavenack_r<=16'h2;//i2cslavew0r1 ? 16'h4 : 16'h2;
		7'h68: i2cslavenack_r<=16'h2;
		7'h5d: i2cslavenack_r<=16'h2;
		7'h1c: i2cslavenack_r<=16'h3;
		default: i2cslavenack_r<=16'h3;
	endcase
end


assign hw.vc707.sysclk=sysclk;
assign hw.vc707.sgmiiclk_q0_p=sgmiiclk;
assign hw.vc707.sgmiiclk_q0_n=~sgmiiclk;
assign hw.vc707.si5324_out_c_p=si5324clk;
assign hw.vc707.si5324_out_c_n=~si5324clk;
assign hw.vc707.pcie.clk_qo_p=1'b0;
assign hw.vc707.pcie.clk_qo_n=1'b1;
assign hw.vc707.sma_mgt_rx_p=hw.vc707.sfp.tx_p;
assign hw.vc707.sma_mgt_rx_n=hw.vc707.sfp.tx_n;
assign hw.vc707.sfp.rx_n=hw.vc707.sma_mgt_tx_n;
assign hw.vc707.sfp.rx_p=hw.vc707.sma_mgt_tx_p;
assign hw.vc707.sfp.los=1'b0;
assign hw.vc707.gpio_sw_c=1'b1;
assign hw.fmc1.llmk_dclkout_2=clk250;
assign hw.fmc1.llmk_sclkout_3=clk500cnt[12];
assign hw.fmc1.lmk_dclk8_m2c_to_fpga=clk500;
assign hw.fmc1.lmk_dclk10_m2c_to_fpga=clk125;
assign si5324clk=hw.vc707.rec_clock;
assign hw.fmc2.llmk_dclkout_2=clk250;
assign hw.fmc2.llmk_sclkout_3=clk500cnt[12];
assign hw.fmc2.lmk_dclk8_m2c_to_fpga=clk500;
assign hw.fmc2.lmk_dclk10_m2c_to_fpga=clk125_delay;
//assign hw.vc707.usb2uart.tx=0;
//assign hw.vc707.usb2uart.rx=0;
localparam BAUD=9600000;
qubic #(.BAUD(BAUD),.SIM(SIM)) qubic(.fpga(fpga));

reg [31:0] sysclkcnt=0;
always @(posedge sysclk) begin
	sysclkcnt<=sysclkcnt+1;
end
reg [31:0] sgmiiclkcnt=0;
always @(posedge sgmiiclk) begin
	sgmiiclkcnt<=sgmiiclkcnt+1;
end
wire uartclk;
reg [31:0] uartclkcnt=0;
always @(posedge uartclk) begin
	uartclkcnt<=uartclkcnt+1;
end
wire [7:0] rxdata;
wire [7:0]  txdata;//=sysclkcnt[20:13];//8'hcc;
wire rxvalid;
wire txready;
wire txstart;//sysclkcnt[13:0]==100;
reg txready_d=0;
reg txready_d2=0;
assign uartclk=qubic.qubichw_config.uartclk;
uart #(.DWIDTH(8),.NSTOP(1),.UARTCLK(100000000),.BAUD(BAUD))
uart (.clk(uartclk)
,.TX(hw.vc707.usb2uart.tx)
,.RX(hw.vc707.usb2uart.rx)
//,.rst(uartclkcnt<4)
,.rst(1'b0)
,.txdata(txdata)
,.txstart(txstart)
,.rxdata(rxdata)
,.rxvalid(rxvalid)
,.txready(txready)
);
localparam LINES=10;
localparam TOTALBYTE=LINES*8;
reg  [LINES*64-1:0] cmdsim={8'hff,24'hffffff,32'hffffffff
,8'hff,24'hffffff,32'hffffff00
,64'h0100001700e02281
,64'h0100001800000001
,64'h010000090000e920
,64'h0000000900000000
,64'h0100000a00000013
,64'h00000000facefeed
,64'h00000001deadbeef
,64'h0100001500000000
};
/*localparam TOTALBYTE=LINES*8-1;
	{8'd1,24'h0,32'habcd1234
,8'd0,24'h0,32'hfacefeed
,8'd0,24'h1,24'h0
,8'hff,24'hffffff,32'hffffffff
,8'hff,24'hffffff,32'hffffff00
,8'd0,24'h1,32'hdeadbeef
,8'd0,24'h3,32'hfacefeed
};
*/
reg [7:0] txdata_r=0;
reg [31:0] txcnt=0;
always @(posedge uartclk) begin
	txready_d<=txready;
	if (txstart) begin
		{txdata_r,cmdsim}=cmdsim<<8;
		txcnt<=txcnt+1;
	end
end
assign txdata=cmdsim[LINES*64-1:LINES*64-8];
assign txstart=(txready_d&~txready_d2&(txcnt<=TOTALBYTE) )|uartclkcnt==4;
assign hw.vc707.VP_0=1'b0;
assign hw.vc707.VN_0=1'b1;
//assign hw.vc707.sgmii_rx_p=hw.vc707.sgmii_tx_p;
//assign hw.vc707.sgmii_rx_n=hw.vc707.sgmii_tx_n;

wire mdio_i,mdio_o,mdio_t;
assign mdio_t=qubic.qubichw_config.mdio_t;
assign mdio_i=mdio_o;
IOBUF mdiobuf(.O(mdio_o),.I(mdio_i),.T(~mdio_t),.IO(hw.vc707.phy_mdio));

wire resetdone;
gmii ifgmii();
sgmii_ethernet_pcs_pma #(.SIM(SIM))
sgmii_ethernet_pcs_pma(.gtrefclk(sgmiiclk)
,.rxn(hw.vc707.sgmii_tx_n)
,.rxp(hw.vc707.sgmii_tx_p)
,.txn(hw.vc707.sgmii_rx_n)
,.txp(hw.vc707.sgmii_rx_p)
,.gmii(ifgmii.phy)
,.independent_clock_bufg(sysclk)
,.reset(sysclkcnt==20)//hwreset)
,.resetdone(resetdone)
,.status_vector()
);
reg tx_en=0;
always @(posedge ifgmii.tx_clk) begin
	tx_en<=sysclkcnt>32'h1000;
end
/*assign gmii.tx_en=tx_en;
assign gmii.txd=8'hde;
assign gmii.tx_er=1'b0;*/
// ping example
/*localparam NBYTES=14*8-2;
localparam data={
64'h55555555555555d5
,64'h00105ad155b2c46e
,64'h1f01d90d08004500
,64'h0054a66240004001
,64'h0f4ec0a801c8c0a8
,64'h01e00800b0285ae8
,64'h00936b39235f0000
,64'h00009af004000000
,64'h0000101112131415
,64'h161718191a1b1c1d
,64'h1e1f202122232425
,64'h262728292a2b2c2d
,64'h2e2f303132333435
,48'h36370cb55572
};*/
// arp example
/*localparam NBYTES=9*8;
localparam data={
64'h55555555555555d5
,64'hffffffffffffc46e
,64'h1f01d90d08060001
,64'h080006040001c46e
,64'h1f01d90dc0a801c8
,64'h000000000000c0a8
,64'h01e0000000000000
,64'h0000000000000000
,64'h0000000072bda56a
};*/
// udplbw
localparam NBYTES=72;
localparam data={
64'h55555555555555d5
,64'h515542494301c46e
,64'h1f01d90d08004500
,64'h002c449040004011
,64'h7138c0a801c8c0a8
,64'h01e089f9d0030018
,64'h00c7100000000000
,64'h0000100000010000
,64'h00000000e2f1f1d4
};
/*localparam NBYTES=72;
localparam data={
64'h55555555555555d5
,64'h515542494301c46e
,64'h1f01d90d08004500
,64'h0024d1e940004011
,64'he3e6c0a801c8c0a8
,64'h01e0b461d0030010
,64'he66d100000020000
,64'h0000000000000000
,64'h00000000f7eeb878
};
*/
/*
64'h55555555555555d5
,64'h515542494301c46e
,64'h1f01d90d08004500
,64'h003c83ed40004011
,64'h31cbc0a801c8c0a8
,64'h01e0c7d9d0030028
,64'h164684b678b2829d
,64'h4979000000000000
,64'h00ff000000020000
,64'h00ff000000030000
,64'h00fff0b2aa58
};
localparam NBYTES=8*10+6;
localparam data={
64'h55555555555555d5
,64'h515542494301c46e
,64'h1f01d90d08004500
,64'h003c843b40004011
,64'h317dc0a801c8c0a8
,64'h01e09683d0030028
,64'h1a9984b678b2829d
,64'h4979100000000000
,64'h0000100000020000
,64'h0000100000030000
,64'h0000f0c07ab2
};
	64'h55555555555555d5
,64'h515542494301c46e
,64'h1f01d90d08004500
,64'h003cd40d40004011
,64'he1aac0a801c8c0a8
,64'h01e0b6c3d0030028
,64'h8ca884b678b2829d
,64'h497900000000dead
,64'hbeef000000020000
,64'h000a000000020000
,48'h000a182eac34
};

localparam NBYTES=8*9;
localparam data={
64'h55555555555555d5
,64'hf7ffffffffffcc66
,64'h5f0d914d08080544
,64'h0024068140400051
,64'hb0a7c8e00980c8e0
,64'hbfffecc60db64810
,64'hb3f2144346720400
,64'h0000000000000000
,64'h000000003755c412
};
*/
assign ifgmii.tx_clk=ifgmii.rx_clk;
reg [6:0] inc=0;
reg [31:0] txclkcnt=0;
//wire ethstart= txclkcnt[7]&(txclkcnt[6:0]==inc) && resetdone;
//reg ethstart_d=0;
//reg [8*NBYTES-1:0] datasr=0;
always @(posedge ifgmii.tx_clk) begin
	txclkcnt<=txclkcnt+1;
/*	ethstart_d<=ethstart;
	if (ethstart&~ethstart_d) begin
		datasr<=data;
		inc<=inc+1;
	end
	if (|datasr)
		datasr<= datasr<<8;
	*/
end
//assign gmii.tx_en= |datasr;
//assign gmii.tx_er= 1'b0;
//assign gmii.txd=datasr[8*NBYTES-1:8*NBYTES-8];

reg simdv=0;
localparam MAXNBYTES=200*8;
reg [8*MAXNBYTES-1:0] datasr=0;
`include "simin.vh"
assign ifgmii.tx_en= simdv;
assign ifgmii.tx_er= 1'b0;
assign ifgmii.txd=datasr[8*MAXNBYTES-1:8*MAXNBYTES-8];


wire trig100=sgmiiclkcnt[31:1]==100;
reg trig100_r=0;
wire [11:0] addr;
wire [31:0] wdata;
wire [31:0] rdata;
wire rdatavalid;
wire start;
wire w0r1;
wire busy;
reg start_d=0;

//assign qubic.lbreg.lb.wcmd={8'h03,24'd256,32'h0};
//assign qubic.lbreg.lb.wvalid=1'b1;
//always @(posedge qubic.lbreg.lb.clk) begin
//begin qubic.qubichw_config.udplb64.lbrxdata_r<={8'h03,24'd256,32'h0}; qubic.qubichw_config.udplb64.lbrxdv_r<=1'b1; end
//end
/*always @(posedge qubic.lbreg.lb.clk) begin

if (1'b0&start) begin
	//lbrxdata_r<=lbrxdatafifo;
	//lbrxdv_r<=lbrxdvfifo;
//	qubic.qubichw_config.udplb64.lbrxdata_r={8'h00,24'd32,32'h12345679};
//	qubic.qubichw_config.udplb64.lbrxdv_r=1'b1;
	qubic.lbreg.reg_axifmc1adc0_addr=addr;
	qubic.lbreg.reg_axifmc1adc0_wdata=wdata;
	qubic.lbreg.reg_axifmc1adc0_w0r1=w0r1;
	qubic.lbreg.reg_axifmc1adc0_start=start;
	qubic.qubichw_config.udplb64.lbrxdata_r={8'h00,24'd32,32'h0};
	qubic.qubichw_config.udplb64.lbrxdv_r=1'b1;
end
else begin qubic.qubichw_config.udplb64.lbrxdata_r<={8'h03,24'd256,32'h0}; qubic.qubichw_config.udplb64.lbrxdv_r<=1'b0; end
end
assign busy=qubic.qubichw_config.lb_axi4lite_fmc1_adc0.busy;
//assign qubic.qubichw_config.udplb64.lbrxdatafifo= trig100 ? {8'h00,24'd32,32'h12345679} : 0;
//assign qubic.qubichw_config.udplb64.lbrxdvfifo= trig100 ? 1'b1 :0;


//localparam LINE=4000;
localparam LINE=14;
localparam W0R1ADDRDATA={1'b1,12'h4,32'h1
,1'b0,12'h8,32'h1
,1'b1,12'h4,32'h1
,1'b0,12'h20,32'h7
,1'b1,12'h4,32'h1
,1'b0,12'h2c,32'h0
,1'b1,12'h4,32'h1
,1'b0,12'h2c,32'h0
,1'b1,12'h4,32'h1
,1'b0,12'h4,32'h1
,{(LINE-14){1'b1,12'h4,32'h1}}
};
reg [15:0] cnt=0;
reg busy_d=0;
always @(posedge sgmiiclk) begin
    busy_d<=busy;
    if (~busy&busy_d & cnt<LINE & (sgmiiclkcnt>100))
        cnt<=cnt+1;
end
localparam WADWIDTH=1+12+32;
assign {w0r1,addr,wdata}=W0R1ADDRDATA[(LINE-1-cnt)*WADWIDTH+:WADWIDTH];
assign start=~busy & cnt<LINE-1;
*/

endmodule
