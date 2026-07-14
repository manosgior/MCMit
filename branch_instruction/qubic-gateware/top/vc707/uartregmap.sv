interface iuart_regmap#(parameter LBCWIDTH=8,parameter LBAWIDTH=24,parameter LBDWIDTH=32,parameter WRITECMD=1,parameter READCMD=0)();

ilocalbus #(.LBCWIDTH(LBCWIDTH),.LBAWIDTH(LBAWIDTH),.LBDWIDTH(LBDWIDTH),.READCMD(READCMD),.WRITECMD(WRITECMD))
lb();


reg lbrready_r=0;
reg lbwvalid_r=0;
reg [LBCWIDTH-1:0] lbrcmd_r=0;
reg lbrready_r1=0;
reg lbrready_r2=0;

`include "uartdefine.vh"
always @(posedge lb.clk) begin
	//lbwvalid_r<=lb.wvalid;
	lbrready_r2<=lbrready_r1;
	lbrready_r1<=lbrready_r;
	case (lb.wctrl)
		lb.writecmd: begin
			lbrready_r<=lb.wvalid;
			lb.rctrl<=lb.wctrl;
			lb.raddr<=lb.waddr;
			lb.rdata<=lb.wdata;
			`include "uartwrite.vh"
		end
		lb.readcmd: begin
			lbrready_r<=lb.wvalid;
			lb.rctrl<=lb.wctrl;
			lb.raddr<=lb.waddr;
			casex (lb.waddr)
				`include "uartread.vh"
				default: lb.rdata<= 32'hdeadbeef;
			endcase
		end
		default: begin
			lbrready_r<=lb.wvalid;
			lb.rctrl<=lb.wctrl;
			lb.raddr<=lb.waddr;
			lb.rdata<=lb.wdata;
		end
	endcase
end
assign lb.rready= lbrready_r2;

//endinterface

modport cfg(
input uartmode,stb_i2cstart,i2cstart,i2cdatatx,clk4ratio,i2cmux_reset_b,hwreset
,output i2cdatarx,i2crxvalid,hwresetstatus,macmsb24,maclsb24,ipaddr);

endinterface

