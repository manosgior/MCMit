`timescale 1ns / 100ps
module loop_tb();
initial begin
	$dumpfile("loop.vcd");
	$dumpvars(17,loop_tb);
end

reg sysclk=0;
initial begin
    while (1) begin
        sysclk=0; #2.5;
        sysclk=1; #2.5;
    end
end

reg sgmiiclk=0;
initial begin
    forever #(4) sgmiiclk=~sgmiiclk;
end

reg [31:0] sysclkcnt=0;
always @(posedge sysclk) begin
	sysclkcnt<=sysclkcnt+1'b1;
end


wire signal_detect=1'b1;
wire speed_is_100=1'b0;
wire speed_is_10_100=1'b0;
wire [4:0] configuration_vector=5'b00000;
wire  reset=sysclkcnt<250;//32'h100;//rxvalid;//sysclkcnt==101;
wire  rxn;
wire  rxp;
wire  txn;
wire  txp;
wire  rxn2;
wire  rxp2;
wire  txn2;
wire  txp2;
wire resetdone1;
wire userclk2_1;
wire sgmii_clk_r1;
wire sgmii_clk_f1;
gmii gmii();
gmii gmii2();
wire resetdone2;
wire userclk2_2;
wire sgmii_clk_r2;
wire sgmii_clk_f2;

sgmii_ethernet_pcs_pma #(.EXAMPLE_SIMULATION(1))
sgmii_ethernet_pcs_pma_1(.gtrefclk(sgmiiclk)
,.rxn(rxn),.rxp(rxp),.txn(txn),.txp(txp)
,.gmii(gmii)
,.independent_clock_bufg(sysclk)
,.reset(reset)
,.resetdone(resetdone1)
);

/*
gig_ethernet_pcs_pma_0 #(.EXAMPLE_SIMULATION(1))
gig_ethernet_pcs_pma_1(
.configuration_vector,.signal_detect,.speed_is_100,.speed_is_10_100
,.gtrefclk_n(~sgmiiclk),.gtrefclk_p(sgmiiclk),.independent_clock_bufg(sysclk)
,.gmii_isolate(gmii.isolate),.gmii_rx_dv(gmii.rx_dv),.gmii_rx_er(gmii.rx_er),.gmii_rxd(gmii.rxd),.gmii_tx_en(gmii.tx_en),.gmii_tx_er(gmii.tx_er),.gmii_txd(gmii.txd),.sgmii_clk_f(sgmii_clk_f1),.sgmii_clk_r(sgmii_clk_r1)
,.rxn(rxn),.rxp(rxp),.txn(txn),.txp(txp)
,.userclk2_out(userclk2_1)
,.userclk_out(gmii.tx_clk_62_5)
,.gt0_qplloutclk_out(),.gt0_qplloutrefclk_out(),.gtrefclk_bufg_out(),.gtrefclk_out(),.mmcm_locked_out(),.pma_reset_out(),.rxuserclk2_out(),.rxuserclk_out(),.sgmii_clk_en(),.status_vector()
,.reset(reset),.resetdone(resetdone1)
);
*/
loop_gig #(.EXAMPLE_SIMULATION(1))
gig_ethernet_pcs_pma_2(
.configuration_vector,.signal_detect,.speed_is_100,.speed_is_10_100
,.gtrefclk_n(~sgmiiclk),.gtrefclk_p(sgmiiclk),.independent_clock_bufg(sysclk)
,.gmii_isolate(gmii2.isolate),.gmii_rx_dv(gmii2.rx_dv),.gmii_rx_er(gmii2.rx_er),.gmii_rxd(gmii2.rxd),.gmii_tx_en(gmii2.tx_en),.gmii_tx_er(gmii2.tx_er),.gmii_txd(gmii2.txd),.sgmii_clk_f(sgmii_clk_f2),.sgmii_clk_r(sgmii_clk_r2)
,.rxn(rxn2),.rxp(rxp2),.txn(txn2),.txp(txp2)
,.userclk2_out(userclk2_2)
,.userclk_out(gmii.tx_clk_62_5)
,.gt0_qplloutclk_out(),.gt0_qplloutrefclk_out(),.gtrefclk_bufg_out(),.gtrefclk_out(),.mmcm_locked_out(),.pma_reset_out(),.rxuserclk2_out(),.rxuserclk_out(),.sgmii_clk_en(),.status_vector()
,.reset(reset),.resetdone(resetdone2)
);
//,.mdc(mdc),.mdio_i(mdio_i1),.mdio_o(mdio_o1),.mdio_t(mdio_t1),.phyaddr
//,.configuration_valid
ODDR gmiiclk1(.Q(gmii.rx_clk),.CE(1'b1),.R(1'b0),.S(1'b0),.C(userclk2_1),.D1(sgmii_clk_r1),.D2(sgmii_clk_f1));
ODDR gmiiclk2(.Q(gmii2.rx_clk),.CE(1'b1),.R(1'b0),.S(1'b0),.C(userclk2_2),.D1(sgmii_clk_r2),.D2(sgmii_clk_f2));
/*stimulus_tb stimulus2
(
//.txp(txp2),.txn(txn2),.rxp(rxp2),.rxn(rxn2),

.gmii_tx_clk(),
.gmii_rx_clk(gmii2.rx_clk),
.gmii_txd(gmii2.txd),
.gmii_tx_en(gmii2.tx_en),
.gmii_tx_er(gmii2.tx_er),
.gmii_rxd(gmii2.rxd),
.gmii_rx_dv(gmii2.rx_dv),
.gmii_rx_er(gmii2.rx_er),

.speed_is_10_100(speed_is_10_100),
.speed_is_100(speed_is_100),

.configuration_finished(resetdone2),
.tx_monitor_finished(),
.rx_monitor_finished()
);*/
assign rxp=txp2;
assign rxn=txn2;
assign rxp2=txp;
assign rxn2=txn;
/*stimulus_tb stimulus
(
//.txp(txp),.txn(txn),.rxp(rxp),.rxn(rxn),

.gmii_tx_clk(),
.gmii_rx_clk(gmii.rx_clk),
.gmii_txd(gmii.txd),
.gmii_tx_en(),//gmii.tx_en),
.gmii_tx_er(gmii.tx_er),
.gmii_rxd(gmii.rxd),
.gmii_rx_dv(gmii.rx_dv),
.gmii_rx_er(gmii.rx_er),

.speed_is_10_100(speed_is_10_100),
.speed_is_100(speed_is_100),

.configuration_finished(resetdone1),
.tx_monitor_finished(),
.rx_monitor_finished()
);*/
localparam DATALEN_64=3;
localparam data={64'h55555555555555d5
,64'hda02030405065a02
,64'h03040506002e0102
/*,64'h030405060708090a
,64'h0b0c0d0e0f101112
,64'h131415161718191a
,64'h1b1c1d1e1f202122
,64'h232425262728292a
,64'h2b2c2d2e1419d1dd
*/
};
assign gmii.tx_clk=gmii.rx_clk;
wire txstart= sysclkcnt[6] && resetdone1;
reg txstart_d=0;
reg [64*3-1:0] datasr=0;
always @(posedge gmii.tx_clk) begin
	txstart_d<=txstart;
	if (txstart&~txstart_d)
		datasr<=data;
	if (|datasr)
		datasr<= datasr<<8;
end
assign gmii.tx_en= |datasr;
assign gmii.tx_er= 1'b0;
assign gmii.txd=datasr[64*DATALEN_64-1:64*DATALEN_64-8];
assign gmii2.tx_en= |datasr;
assign gmii2.tx_er= 1'b0;
assign gmii2.txd=datasr[64*DATALEN_64-1:64*DATALEN_64-8];
endmodule
/*
wire userclk2_2;
gig_ethernet_pcs_pma_0 #(.EXAMPLE_SIMULATION(1))gig_ethernet_pcs_pma_1(
.independent_clock_bufg(sysclk)
//,.configuration_valid
,.configuration_vector,.signal_detect,.speed_is_100,.speed_is_10_100
,.reset(reset2)
,.gmii_isolate(gmii2.isolate),.gmii_rx_dv(gmii2.rx_dv),.gmii_rx_er(gmii2.rx_er),.gmii_rxd(gmii2.rxd),.gmii_tx_en(gmii2.tx_en),.gmii_tx_er(gmii2.tx_er),.gmii_txd(gmii2.txd)
,.userclk2_out(userclk2_2),.userclk_out(gmii2.tx_clk_62_5),.rxuserclk_out(gmii2.rx_clk_62_5)
,.gtrefclk_n(~sgmiiclk),.gtrefclk_p(sgmiiclk)
,.rxn(txn),.rxp(txp),.txn(rxn),.txp(rxp)
//,.mdc(mdc),.mdio_i(mdio_o),.mdio_o(),.mdio_t(mdio_t),.phyaddr
,.status_vector(),.mmcm_locked_out(),.pma_reset_out(),.resetdone(),.rxuserclk2_out(),.sgmii_clk_en(gmii2.clk_en),.sgmii_clk_f(sgmii_clk_f_2),.sgmii_clk_r(sgmii_clk_r_2),.gt0_qplloutclk_out(),.gt0_qplloutrefclk_out(),.gtrefclk_bufg_out(),.gtrefclk_out()
);*/
//ODDR gmiiclk2(.Q(gmii2.rx_clk),.CE (1'b1),.R  (1'b0),.S  (1'b0),.C  (userclk2_2),.D1 (sgmii_clk_r_2),.D2 (sgmii_clk_f_2));
//assign gmii2.tx_en= gmii2.txcnt[3] && (gmii2.txcnt>203) ;//&& mdio_t;
//assign gmii.tx_er=1'b0;
//assign gmii2.txd=gmii2.testdata;
//assign gmii.txd=gmii.testdata;
//assign gmii2.tx_er=1'b0;
//assign mdc=sysclkcnt[4];
/*localparam data={64'h55555555555555d5
,64'hda02030405065a02
,64'h03040506002e0102
,64'h030405060708090a
,64'h0b0c0d0e0f101112
,64'h131415161718191a
,64'h1b1c1d1e1f202122
,64'h232425262728292a
,64'h2b2c2d2e1419d1dd
};
reg [31:0] rxvalidcnt=0;
reg mdiodone=0;
wire txstart=gmii.txcnt[5] &&( gmii.txcnt>200) && mdiodone;
reg [64*9-1:0] datasr=0;
always @(posedge gmii.tx_clk) begin
	if (txstart)
		datasr<=data;
	if (|datasr)
		datasr<= datasr<<8;
end
always @(posedge sysclk) begin
	if (rxvalid&~first)
		mdiodone<=1'b1;
end
assign gmii.tx_en= |datasr;
assign gmii.txd=datasr[64*9-1:64*9-8];
*/

/*wire [4:0] phyaddr=5'b00001;
wire configuration_valid=1'b1;


wire rxvalid;
reg first=1'b1;
reg rxvalid_d=0;
wire  reset2=sysclkcnt<32'h100;//rxvalid;//sysclkcnt==201;
wire mdc;
wire mdio_im;
wire mdio_om;
wire mdio_tm;
wire mdio;
reg resetdone1_d=0;
mdiomaster mdiomaster(.clk(sysclk)
,.busy()
,.clk4ratio(10)
,.datarx(mdiodatarx)
,.datatx(16'h8100)//16'h0100)
,.mdc(mdc)
,.mdio_i(mdio_im)
,.mdio_o(mdio_om)
,.mdio_t(mdio_tm)
,.opr1w0(~first)
,.phyaddr
,.regaddr(0)
,.rst(1'b0)
,.rxvalid(rxvalid)
,.start((resetdone1 & ~ resetdone1_d)|| (first&rxvalid))
);
IOBUF mdiobufmaster(.O(mdio_im),.I(mdio_om),.T(mdio_tm),.IO(mdio));
wire mdio_t1;
wire mdio_i1;
wire mdio_o1;
wire mdio_t2;
wire mdio_i2;
wire mdio_o2;
IOBUF mdiobufslave(.O(mdio_i1),.I(mdio_o1),.T(mdio_t1),.IO(mdio));
always @(posedge sysclk) begin
	resetdone1_d<=resetdone1;
	rxvalid_d<=rxvalid;
	if (rxvalid & first)
		first<=1'b0;
end
*/
//sysclkcnt==101));
