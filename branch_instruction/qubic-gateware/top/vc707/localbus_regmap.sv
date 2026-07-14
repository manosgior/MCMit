interface ilocalbus_regmap#(parameter LBCWIDTH=8,parameter LBAWIDTH=24,parameter LBDWIDTH=32,parameter WRITECMD=1,parameter READCMD=0)();

ilocalbus #(.LBCWIDTH(LBCWIDTH),.LBAWIDTH(LBAWIDTH),.LBDWIDTH(LBDWIDTH),.READCMD(READCMD),.WRITECMD(WRITECMD))
lb();


reg lbrready_r=0;
reg lbwvalid_r=0;
reg [LBCWIDTH-1:0] lbrcmd_r=0;
reg lbrready_r1=0;
reg lbrready_r2=0;
reg rdbusy=0;
//reg [63:0] rcmd64=0;
`include "lbdefine.sv"
always @(posedge lb.clk) begin
	//lbwvalid_r<=lb.wvalid;
	lbrready_r2<=lbrready_r1;
	rdbusy<=lb.rdbusy;
	case (lb.wctrl)
		lb.writecmd: begin
			lbrready_r1<=lbrready_r;
			lbrready_r<=lb.wvalid;
			lb.rctrl<=lb.wctrl;
			lb.raddr<=lb.waddr;
			lb.rdata<=lb.wdata;
			`include "lbwrite.sv"
		end
		lb.readcmd: begin
			lbrready_r1<=lbrready_r;
			lbrready_r<=lb.wvalid;
			lb.rctrl<=lb.wctrl;
			lb.raddr<=lb.waddr;
			casex (lb.waddr)
				`include "lbread.sv"
				default: lb.rdata<= 32'hdeadbeef;
			endcase
		end
/*		lb.blockread: begin   // developing, not working yet
			if (~rdbusy) begin
				lbrready_r1<=lb.wvalid;
				lb.rctrl<=lb.wctrl;
				lb.raddr<=lb.waddr;
//				casex (lb.waddr)
//					24'd256: begin lb.rdata<=0; end
//				endcase
//			endcase
            end
			else begin
				casex (lb.waddr)
					24'd256: begin rcmd64<=cntbuf_buf.buf0.rd.data; lbrready_r1<=cntbuf_buf.buf0.rd.dv; end
				endcase
				{lb.rctrl,lb.raddr,lb.rdata}<=rcmd64;
			end

		end
		*/
		default: begin
			lbrready_r<=lb.wvalid;
			lb.rctrl<=lb.wctrl;
			lb.raddr<=lb.waddr;
			lb.rdata<=lb.wdata;
		end
	endcase
end
assign lb.rready= lbrready_r2;
assign lb.wen=1'b1;
/*lbbuf #(.AWW(5),.DWW(8),.DWR(64)) cntbuf_buf(1'b0);
assign cntbuf_buf.buf0.rd.clk=lb.clk;
assign cntbuf_buf.stb_read=lb.wctrl==lb.blockread & lb.wvalid & lb.waddr==24'd256;
assign lb.rdbusy=|{cntbuf_buf.rdbusy};
*/
//wire stb_bufreadtest=lb.wctrl==lb.readcmd&lb.wvalid&lb.waddr[23:10]==14'h1;
//bufread #(.AW(10),.DW(16)) bufreadtest(.rclk(lb.clk),.ren(stb_bufreadtest));
//assign bufreadtest.rd.addr=lb.waddr[9:0];
//{12'h1,{10{1'bx}}}: lb.rdata<= bufreadtest.rd.data;

//endinterface

modport cfg(
input test,test2,err,uartmode,xadcupdate,stb_i2cstart,i2cstart,i2cdatatx,clk4ratio,i2cmux_reset_b,fmcdacen,sfptesttx,stb_hwreset,sfptxdisable,mdiodatatx,stb_mdiostart,mdioclk4ratio
,axifmc1adc0_addr,axifmc1adc0_w0r1,axifmc1adc0_wdata,axifmc1adc1_addr,axifmc1adc1_w0r1,axifmc1adc1_wdata,axifmc1dac_addr,axifmc1dac_w0r1,axifmc1dac_wdata,axifmc2adc0_addr,axifmc2adc0_w0r1,axifmc2adc0_wdata,axifmc2adc1_addr,axifmc2adc1_w0r1,axifmc2adc1_wdata,axifmc2dac_addr,axifmc2dac_w0r1,axifmc2dac_wdata,stb_axifmc1adc0_start,stb_axifmc1adc1_start,stb_axifmc1dac_start,stb_axifmc2adc0_start,stb_axifmc2adc1_start,stb_axifmc2dac_start,axifmc1adc0_start,axifmc1adc1_start,axifmc1dac_start,axifmc2adc0_start,axifmc2adc1_start,axifmc2dac_start,si5324_rst,countperrequest,stb_bufreadtestreset
,output xadctemp,xadcaux4,xadcaux12,i2cdatarx,i2crxvalid,fmcprsnt,fmcpgm2c,sfptestrx,hwresetstatus,mdiodatarx,mdiorxvalid,axifmc1adc0_rdata,axifmc1adc0_rdatavalid,axifmc1adc1_rdata,axifmc1adc1_rdatavalid,axifmc1dac_rdata,axifmc1dac_rdatavalid,axifmc2adc0_rdata,axifmc2adc0_rdatavalid,axifmc2adc1_rdata,axifmc2adc1_rdatavalid,axifmc2dac_rdata,axifmc2dac_rdatavalid,macmsb24,maclsb24,ipaddr,bufreadtestfull
,freq_lb,freq_sgmiiclk,freq_sma_mgt_refclk,freq_si5324_out_c,freq_pcie_clk_qo,freq_user_clock,freq_fmc1_llmk_dclkout_2,freq_fmc1_llmk_sclkout_3,freq_fmc1_lmk_dclk8_m2c_to_fpga,freq_fmc1_lmk_dclk10_m2c_to_fpga,freq_fmc2_llmk_dclkout_2,freq_fmc2_llmk_sclkout_3,freq_fmc2_lmk_dclk8_m2c_to_fpga,freq_fmc2_lmk_dclk10_m2c_to_fpga,freq_rxusrclk_sfp,freq_txusrclk_sfp
);
modport dsp(input test,test2,err
,output test1
);
endinterface

//interface localbus#(parameter LBCWIDTH=8,parameter LBAWIDTH=24,parameter LBDWIDTH=32)();

