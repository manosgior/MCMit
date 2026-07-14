`timescale 1ns / 1ns
module bufread #(parameter DWW=8
	,parameter DWR=32
	,parameter AWW=8
	,parameter SIM=0
	,localparam AWR=DWW>DWR ? (AWW+$clog2(DWW/DWR)) : (AWW-$clog2(DWR/DWW))
	,localparam SZR=(32'b1<<AWR)-1
	)
	(
//	ibufio.source rd
//	,ibufio.destin wr
	input wclk
	,input rclk
	,input [DWW-1:0] wdata
	,input [AWW-1:0] waddr
	,input wen
	,input ren
	,input [AWR-1:0] raddr
	,output rdv
	,output [DWR-1:0] rdata
	,output full
	,input reset
);

	reg [AWW-1:0] wraddr=0;
	reg [AWW-1:0] wraddr_next=1;
	reg wren=1;
	reg full_r=0;//&waddr;
	wire wrclk=wclk;
	reg [DWW-1:0] wrdata=0;
	dpram2 #(.DWW(DWW),.DWR(DWR),.AWW(AWW),.SIM(SIM),.BUFIN(1),.BUFOUT(1))
	dpram(.wrclk(wrclk),.rdclk(rclk),.wraddr(wraddr),.wrdata(wrdata),.wren(wren),.rden(ren),.rddata(rdata),.rddv(rdv),.rdaddr(raddr));

	always @(posedge wclk or posedge reset) begin
		wrdata<=wdata;
		if (reset) begin
			wraddr<= 0;
			wraddr_next<= 1;
			full_r<=0;
			wren<=1'b1;
		end
		else if (wen & (~full_r)) begin
			wraddr<= wraddr +1;
			wraddr_next<= wraddr_next +1;
			wren<=1'b1;
			full_r<= &(wraddr_next);
		end
		else begin
			wren<=1'b0;
		end
	end
	assign full=full_r;
endmodule
