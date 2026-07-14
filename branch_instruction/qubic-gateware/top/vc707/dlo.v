module dlo #(parameter WIDTH=16,NSTAGE=16,NORMALIZE=1,BUFIN=0,GW=1,NRIDER=0)(
		input clk
		,input [NSTAGE:0] freq
		,input [WIDTH-1:0] amp
		,input reset
		,output [WIDTH-1:0] xout
		,output [WIDTH-1:0] yout
		);
reg [16:0] phacc=0;
reg [15:0] ampset=0;
always @(posedge clk) begin
    phacc<=reset ? 0 : phacc+freq;
    ampset<=amp;
end
cordicg1 #(.WIDTH(16),.NSTAGE(16),.NORMALIZE(1),.BUFIN(0),.GW(1),.NRIDER(0))
cordicg1(.clk(clk),.opin(1'b0),.xin(ampset),.yin(16'h0),.phasein(phacc),.xout(xout),.yout(yout),.phaseout(),.error(),.gin(1'b1),.gout());
endmodule

