interface ibufio #(parameter DW=8,parameter AW=8)();
	wire clk;
	wire [DW-1:0] data;
	wire [AW-1:0] addr;
	wire dv;
	wire en;
	modport source( output data,dv
	,input clk,addr,en
	);
	modport destin( input data,dv
	,output clk,en,addr
	);
endinterface
