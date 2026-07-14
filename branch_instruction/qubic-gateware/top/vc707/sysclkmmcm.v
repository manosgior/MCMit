module sysclkmmcm(input sysclk
,output clk250
,output clk200
,output clk100
,output clk125
,output [31:0] clk250cnt
,output [31:0] clk200cnt
,output [31:0] clk100cnt
,output [31:0] clk125cnt
,output mmcm_locked
,input mmcm_reset
);
wire mmcm_clk200;
wire mmcm_clk100;
wire mmcm_clk125;
wire mmcmclkfbout,mmcmclkfbin;
MMCME2_BASE #(.BANDWIDTH("OPTIMIZED")
,.CLKIN1_PERIOD(5.0)  // 200MHz
,.CLKFBOUT_MULT_F(5)   //
,.DIVCLK_DIVIDE(1)     //
,.CLKFBOUT_PHASE(0.0)
,.CLKOUT0_DIVIDE_F(10)   // 100M
,.CLKOUT0_DUTY_CYCLE(0.5)
,.CLKOUT0_PHASE(0.0)
,.CLKOUT1_DIVIDE(5)   // 200M
,.CLKOUT1_DUTY_CYCLE(0.5)
,.CLKOUT1_PHASE(0.0)
,.REF_JITTER1(0.0)
,.CLKOUT2_DIVIDE(8)   // 125M
,.CLKOUT2_DUTY_CYCLE(0.5)
,.CLKOUT2_PHASE(0.0)
,.CLKOUT3_DIVIDE(4)   // 250M
,.CLKOUT3_DUTY_CYCLE(0.5)
,.CLKOUT3_PHASE(0.0)
//,.REF_JITTER2(0.0)
,.STARTUP_WAIT("FALSE")
) mmcme2_base_sysclk (.CLKIN1(sysclk)
,.CLKOUT0(mmcm_clk100)
,.CLKOUT1(mmcm_clk200)
,.CLKOUT2(mmcm_clk125)
,.CLKOUT3(mmcm_clk250)
,.LOCKED(mmcm_locked)
,.CLKFBOUT(mmcmclkfbout)
,.CLKFBIN(mmcmclkfbin)
,.PWRDWN(1'b0)
,.RST(mmcm_reset)
);
BUFG mmcmclkfb(.I(mmcmclkfbout),.O(mmcmclkfbin));
BUFG bufgclk100(.I(mmcm_clk100),.O(clk100));
BUFG bufgclk125(.I(mmcm_clk125),.O(clk125));
BUFG bufgclk200(.I(mmcm_clk200),.O(clk200));
BUFG bufgclk250(.I(mmcm_clk250),.O(clk250));
reg [31:0] clk250cnt_r=0;
always @(posedge clk250) begin
	clk250cnt_r<=clk250cnt_r+1;
end
assign clk250cnt=clk250cnt_r;
reg [31:0] clk200cnt_r=0;
always @(posedge clk200) begin
	clk200cnt_r<=clk200cnt_r+1;
end
assign clk200cnt=clk200cnt_r;
reg [31:0] clk100cnt_r=0;
always @(posedge clk100) begin
	clk100cnt_r<=clk100cnt_r+1;
end
assign clk100cnt=clk100cnt_r;
reg [31:0] clk125cnt_r=0;
always @(posedge clk125) begin
	clk125cnt_r<=clk125cnt_r+1;
end
assign clk125cnt=clk125cnt_r;
endmodule
