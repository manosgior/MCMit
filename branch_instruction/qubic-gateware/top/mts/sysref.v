`timescale 1 ns / 1 ps
module sysref
(
input D11
,input E11
,input E10
,input E9
,output clk104_pl_sysref
,output clk104_pl_clk
);

wire _clk104_pl_sysref;
IBUFGDS ibufgds_clk104_pl_sysref(.I(E11),.IB(D11),.O(_clk104_pl_sysref));
BUFG clk104_pl_sysref_bufg(.I(_clk104_pl_sysref),.O(pl_sysref_w));

wire _clk104_pl_clk;
IBUFGDS ibufgds_clk104_pl_clk(.I(E10),.IB(E9),.O(_clk104_pl_clk));
BUFG clk104_pl_clk_bufg(.I(_clk104_pl_clk),.O(pl_clk));


reg pl_sysref_r=0;
always @(posedge pl_clk) begin
	pl_sysref_r<=pl_sysref_w;
end
assign clk104_pl_sysref=pl_sysref_r;
assign clk104_pl_clk=pl_clk;
endmodule
