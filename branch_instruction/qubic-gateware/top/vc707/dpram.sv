`timescale 1ns / 1ns
module dpram2
	#(parameter DWW=8
	,parameter DWR=32
	,parameter AWW=8
	,parameter BUFIN=0
	,parameter BUFOUT=1
	,parameter SIM=0
	,localparam AWR=DWW>DWR ? (AWW+$clog2(DWW/DWR)) : (AWW-$clog2(DWR/DWW))	
	)(
//		ibufio.destin wr
//		,ibufio.source rd
		input wrclk
		,input rdclk
		,input [AWW-1:0] wraddr
		,input [DWW-1:0] wrdata
		,input wren
		,input rden
		,input [AWR-1:0] rdaddr
		,output [DWR-1:0] rddata
		,output rddv


	);
	`define max(a,b) {(a) > (b) ? (a) : (b)}
	`define min(a,b) {(a) < (b) ? (a) : (b)}
	localparam SZW=(32'b1<<AWW)-1;
	localparam SZR=(32'b1<<AWR)-1;
	localparam DWMIN=`min(DWR,DWW);
	localparam SZMAX=`max(SZR,SZW);

	//	localparam SZ=(32'b1<<AWW)-1;
	//	localparam AWR=DWW*AWW/DWR;
	localparam RATIO=DWW>DWR ? DWW/DWR : DWR/DWW;
	localparam WRATIO=$clog2(RATIO);
	reg [DWMIN-1:0] mem[SZMAX:0];
	genvar testi;
	generate
		if (SIM) begin
			wire [DWMIN-1:0] mem0,mem1,mem2,mem3,mem4,mem5,mem6,mem7,mem8,mem9,mema,memb,memc,memd,meme,memf,mem10,mem11,mem12,mem13,mem14,mem15,mem16,mem17,mem18,mem19,mem1a,mem1b,mem1c,mem1d,mem1e,mem1f;
			wire [32*DWMIN-1:0] memtest;
			for (testi=0;testi<32; testi=testi+1)
				assign memtest[(32-testi)*DWMIN-1:(31-testi)*DWMIN]=mem[testi];
			assign {mem0,mem1,mem2,mem3,mem4,mem5,mem6,mem7,mem8,mem9,mema,memb,memc,memd,meme,memf,mem10,mem11,mem12,mem13,mem14,mem15,mem16,mem17,mem18,mem19,mem1a,mem1b,mem1c,mem1d,mem1e,mem1f}=memtest;
		end
	endgenerate

	integer k=0;
	initial begin
		for (k=0;k<SZMAX;k=k+1) begin
			mem[k]=0;
		end
	end

//	ibufio #(.DW(DWW),.AW(AWW)) wr(wclk);
//	ibufio #(.DW(DWR),.AW(AWR)) rd(rclk);
	wire [DWW-1:0] dina_w;
	wire [AWW-1:0] addra_w;
	wire [AWR-1:0] addrb_w;
	wire renb_w;
	wire wena_w;

	if (BUFIN) begin
		reg [AWW-1:0] addra_r=0;
		reg [AWR-1:0] addrb_r=0;
		reg [DWW-1:0] dina_r=0;
		reg wena_r=0;
		reg renb_r=0;
		always @(posedge wrclk) begin
			addra_r<=wraddr;
			wena_r<=wren;
			dina_r<=wrdata;
		end
		always @(posedge rdclk) begin
			addrb_r<=rdaddr;
			renb_r<=rden;
		end
		assign addra_w=addra_r;
		assign wena_w=wena_r;
		assign dina_w=dina_r;
		assign addrb_w=addrb_r;
		assign renb_w=renb_r;

	end
	else begin
		assign addra_w=wraddr;
		assign wena_w=wren;
		assign dina_w=wrdata;
		assign addrb_w=rdaddr;
		assign renb_w=rden;

	end
	wire [DWR-1:0] doutb_w;

	reg rdv=0;
	reg [DWR-1:0] doutb_r=0;
	wire [WRATIO*RATIO-1:0] waddaddr;
	wire [WRATIO*RATIO-1:0] raddaddr;

	genvar i,j;
	generate
	if (DWW <= DWR) begin
		always @(posedge wrclk) begin
			if (wena_w) begin
				mem[addra_w]<=dina_w;
			end
		end
	end
	else begin
		for (i=0;i<RATIO;i++) begin: write
			assign waddaddr[(i+1)*WRATIO-1:i*WRATIO]=i;
			always @(posedge wrclk) begin
				if (wena_w) begin
					mem[{addra_w,waddaddr[(i+1)*WRATIO-1:i*WRATIO]}]=dina_w[(i+1)*DWR-1:i*DWR];
				end
			end
		end
	end
	if (DWW >= DWR) begin
		assign doutb_w=mem[addrb_w];
	end
	else begin
		for (j=0;j<RATIO;j++) begin: read
			assign raddaddr[(j+1)*WRATIO-1:j*WRATIO]=j;
			assign doutb_w[(j+1)*DWMIN-1:j*DWMIN]=mem[{addrb_w,raddaddr[(j+1)*WRATIO-1:j*WRATIO]}];
		end
	end
	endgenerate
	always @(posedge rdclk) begin
		if (renb_w) begin
			doutb_r<=doutb_w;
		end
		rdv<=renb_w;
	end
	assign rddata=BUFOUT ? doutb_r : doutb_w;
	assign rddv=BUFOUT ? rdv : renb_w;
endmodule
