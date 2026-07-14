//`include "constants.vams"
//`include "xc7vx485tffg1761pkg.vh"
`timescale 1ns / 100ps
module gmii_tb(
//gmii ifgmii
);
gmii ifgmii();

reg sysclk=0;
integer cc=0;
initial begin
	$dumpfile("gmii.vcd");
	$dumpvars(17,gmii_tb);
//	for (cc=0; cc<300; cc=cc+1) begin
	while (1) begin
		cc=cc+1;

		sysclk=0; #2.5;
		sysclk=1; #2.5;
	end
	$finish();
end
reg [31:0] sysclkcnt=0;
always @(posedge sysclk) begin
	sysclkcnt<=sysclkcnt+1;
end
reg clk250=0;
initial begin
    forever #(4) clk250=~clk250;
end
reg [31:0] clk250cnt=0;
always @(posedge clk250) begin
	clk250cnt<=clk250cnt+1;
end
reg sgmiiclk=0;
initial begin
    forever #(4) sgmiiclk=~sgmiiclk;
end
assign ifgmii.tx_clk=sgmiiclk;
assign ifgmii.rx_clk=sgmiiclk;

/*assign gmii.tx_en=tx_en;
assign gmii.txd=8'hde;
assign gmii.tx_er=1'b0;*/
// ping example
localparam SIM=1;

reg [6:0] inc=0;
reg [31:0] txclkcnt=0;
wire reset=txclkcnt<100;

always @(posedge ifgmii.tx_clk) begin
   txclkcnt<=txclkcnt+1;
end


//reg [47:0] mac=48'h00105ad155b2;
//reg [47:0] mac=48'h515542494301;
//reg [47:0] mac=48'hc46e1f01d90d;
//reg [47:0] mac=48'h515542494301;//503eaa059701;
//reg [47:0] mac=48'h503eaa059701;
//reg [31:0] ip=32'hc0a801e0;
reg [47:0] mac=48'h55aabbccddee;
reg [31:0] ip=32'hc0a801e0;

localparam MAXNBYTES=200*8;
reg [8*MAXNBYTES-1:0] datasr=0;
reg simdv=0;


`include "simin.vh"
assign ifgmii.rx_dv= simdv;//|datasr;
assign ifgmii.rx_er=1'b0;
assign ifgmii.rxd=datasr[8*MAXNBYTES-1:8*MAXNBYTES-8];


assign ethclk=ifgmii.tx_clk;

udplink ifudp(.reset(ethreset),.clk(ethclk));
gmii2udp #(.SIM(SIM))
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
udpcnt(.clk(ethclk),.udp(ifudpportd002),.reset(ethreset),.udprxerr(udprxerr),.countperrequest(countperrequest));
wire [15:0] txlength;
wire [15:0] rxlength;
ilocalbus #(.LBCWIDTH(8),.LBAWIDTH(24),.LBDWIDTH(32),.WRITECMD(0),.READCMD(8'h10))
udplb();
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
assign udplb.clk=clk250;
assign txlength=rxlength;
regmap#(.LBCWIDTH(8),.LBAWIDTH(24),.LBDWIDTH(32))
lbreg(
);
assign udplb.wen=lbreg.lb.wen;
assign lbreg.lb.clk=udplb.clk;
assign lbreg.lb.wcmd=udplb.wcmd;
assign lbreg.lb.wvalid=udplb.wvalid;
assign udplb.rctrl=lbreg.lb.rctrl;
assign udplb.rdata=lbreg.lb.rdata;
assign udplb.raddr=lbreg.lb.raddr;
assign udplb.rready=lbreg.lb.rready;
assign lbreg.lb.readcmd=udplb.READCMD;
assign lbreg.lb.writecmd=udplb.WRITECMD;
assign lbreg.test1=32'hfff000ab;

assign ethreset=txclkcnt==20; 
endmodule
