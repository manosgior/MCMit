module digimark #(parameter FPGA="REGMULT4"
)(
 input clk
,input [0:0] cstrobe
,input [63:0] command
,output mark
);
reg local_stb;
reg [63:0] local_cmd=0;
always @(posedge clk) begin
	local_stb <= cstrobe;
	if (cstrobe) 
		local_cmd <= command;
end

reg [11:0] start=0;
reg [11:0] length=0;
always @(posedge clk) if (local_stb) begin
	start<= local_cmd[61:50];
	length<= local_cmd[49:38];
end

reg [31:0] clkcnt=0;
wire stop=clkcnt==(start+length);
reg mark_r=0;
reg cstrobe_d=0;
always @(posedge clk) begin
	cstrobe_d<=cstrobe;
	clkcnt <= cstrobe_d ? 0 : stop ? clkcnt : (clkcnt + 1);
	if (clkcnt==start) begin
		mark_r<=1;
	end
	else if (clkcnt==start+length) begin
		mark_r<=0;
	end
end
assign mark=mark_r;
endmodule
