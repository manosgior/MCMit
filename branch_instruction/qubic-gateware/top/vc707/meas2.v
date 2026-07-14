module meas2 #(parameter FPGA="REGMULT4"
,parameter tslice=4
,parameter qbits=4
,parameter aw=10
,parameter dw=16
,parameter DEBUG="true"
// Don't override these
,parameter tslicel=$clog2(tslice)
,parameter qbitl=$clog2(qbits)
)(
(* mark_debug = DEBUG *) input clk
,(* mark_debug = DEBUG *) input signed [16*tslice-1:0] xmeasin
,(* mark_debug = DEBUG *) input signed [16*tslice-1:0] ymeasin
,(* mark_debug = DEBUG *) output signed [16+16-1:0] xacc
,(* mark_debug = DEBUG *) output signed [16+16-1:0] yacc
,(* mark_debug = DEBUG *) input reset
,(* mark_debug = DEBUG *) input wstrobe
,(* mark_debug = DEBUG *) input [aw+tslicel-1:0] waddr
,(* mark_debug = DEBUG *) input [2*dw-1:0] wdata
,(* mark_debug = DEBUG *) input [63:0] command
,(* mark_debug = DEBUG *) input cstrobe
,(* mark_debug = DEBUG *) output active
,(* mark_debug = DEBUG *) output collision
,(* mark_debug = DEBUG *) output signed [16*tslice-1:0] xbase
,(* mark_debug = DEBUG *) output signed [16*tslice-1:0] ybase
,(* mark_debug = DEBUG *) output signed [dw*tslice-1:0] xlo
,(* mark_debug = DEBUG *) output signed [dw*tslice-1:0] ylo
,(* mark_debug = DEBUG *) input signed [31:0] xoffset
,(* mark_debug = DEBUG *) input signed [31:0] yoffset
,(* mark_debug = DEBUG *) input signed [17:0] iqrot
,(* mark_debug = DEBUG *) output resultx
,(* mark_debug = DEBUG *) output resulty
,(* mark_debug = DEBUG *) output done 
);
reg signed [16+16-1:0] xaccr=0;
reg signed [16+16-1:0] yaccr=0;
wire elactive;
wire elactive_w;
wire [dw*tslice-1:0] xlo_w;
wire [dw*tslice-1:0] ylo_w;
element #(.tslice(tslice), .qbits(qbits),
	.aw(aw), .dw(dw)
) el(.clk(clk),
	.command(command), .cstrobe(cstrobe),
	.active(elactive_w), .collision(collision),
	.waddr(waddr), .wdata(wdata), .wstrobe(wstrobe),
	.xout(xlo_w),
	.yout(ylo_w),
	.qsel()
	,.daczero(1'b0)
);
reg [dw*tslice-1:0] xlo_r=0;
reg [dw*tslice-1:0] ylo_r=0;
reg elactive_r1=0;
reg elactive_r2=0;
always @(posedge clk) begin
	xlo_r <= xlo_w;
	ylo_r <= ylo_w;
	elactive_r1 <= elactive_w;
	elactive_r2 <= elactive_r1;
end
assign xlo=xlo_r;
assign ylo=ylo_r;
assign elactive=elactive_r2;
genvar ix;
generate for (ix=0; ix<tslice; ix=ix+1) begin: timeslice
	wire signed [48-1:0] xbase48;
	wire signed [48-1:0] ybase48;
	reg signed [18-1:0] xmeasin_18=0;
	reg signed [18-1:0] ymeasin_18=0;
	reg signed [18-1:0] xlo_18=0;
	reg signed [18-1:0] ylo_18=0;
	always @(posedge clk) begin
		xmeasin_18 <= {xmeasin[ix*dw+:dw],2'b0};
		ymeasin_18 <= {ymeasin[ix*dw+:dw],2'b0};
		xlo_18 <= {xlo[ix*dw+:dw],2'b0};
		ylo_18 <= {-ylo[ix*dw+:dw],2'b0};
	end
	cmultiplier #(.FPGA(FPGA))
	measmult(.clk(clk),.rst(1'b0),.xi(ymeasin_18),.xr(xmeasin_18),.yi(ylo_18),.yr(xlo_18),.zr(xbase48),.zi(ybase48));
	//measmult(.clk(clk),.rst(1'b0),.xi({ymeasin[ix*dw+:dw],2'b0}),.xr({xmeasin[ix*dw+:dw],2'b0}),.yi({-ylo[ix*dw+:dw],2'b0}),.yr({xlo[ix*dw+:dw],2'b0}),.zr(xbase48),.zi(ybase48));  // multiply with xlo-j*ylo complex of conjugate
	assign xbase[ix*dw+:dw]=xbase48[34:19]+xbase48[18];
	assign ybase[ix*dw+:dw]=ybase48[34:19]+ybase48[18];
end
endgenerate
wire [15:0] xbase3,xbase2,xbase1,xbase0;
wire [15:0] ybase3,ybase2,ybase1,ybase0;
assign {xbase3,xbase2,xbase1,xbase0}=xbase;
assign {ybase3,ybase2,ybase1,ybase0}=ybase;
wire signed [16+$clog2(tslice)-1:0] xbase_l, ybase_l;
adder_bank #(.dw(16), .n(tslice)) ax(.clk(clk), .xin(xbase), .xout(xbase_l));
adder_bank #(.dw(16), .n(tslice)) ay(.clk(clk), .xin(ybase), .xout(ybase_l));
wire reset_delay;
wire active_delay;
reg elactive_d=0;
wire active_rising=elactive&~elactive_d;
wire active_faling=~elactive&elactive_d;
wire falingdelay;
reg_delay #(.DW(1),.LEN(4)) delayreset(.clk(clk),.din(active_rising),.dout(reset_delay),.gate(1'b1));
reg_delay #(.DW(2),.LEN(5)) activedelay(.clk(clk),.din({elactive|elactive_d,active_faling}),.dout({active_delay,falingdelay}),.gate(1'b1));
reg resultx_r=0;
reg resulty_r=0;
//wire signed [32-1:0] xbaserot;
//wire signed [32-1:0] ybaserot;
always @(posedge clk) begin
	elactive_d<=elactive;
	xaccr <= reset_delay ? 0 : xaccr+xbase_l;
	yaccr <= reset_delay ? 0 : yaccr+ybase_l;
	if (cordicdone) begin
		resultx_r<=xbaserot[31];
		resulty_r<=ybaserot[31];
	end
end
wire signed [16+16-1:0] xaccr_offset=xaccr-xoffset;
wire signed [16+16-1:0] yaccr_offset=yaccr-yoffset;
//wire activecordic;
//wire cordicdone;
wire signed [32-1:0] xbaserot_w;
wire signed [32-1:0] ybaserot_w;
wire activecordic_w;
wire cordicdone_w;
reg signed [32-1:0] xbaserot=0;
reg signed [32-1:0] ybaserot=0;
reg activecordic=0;
reg cordicdone=0;
cordicg1 #(.WIDTH(32),.NSTAGE(16),.NORMALIZE(1),.BUFIN(1),.GW(2),.NRIDER(0))
//cordicg1(.clk(clk), .opin(1'b0), .xin(xaccr_offset), .yin(yaccr_offset), .phasein({14'b0,iqrot}), .xout(xbaserot), .yout(ybaserot), .phaseout(),.error(),.gin({active_delay,falingdelay}),.gout({activecordic,cordicdone}));
cordicg1(.clk(clk), .opin(1'b0), .xin(xaccr_offset), .yin(yaccr_offset), .phasein({14'b0,iqrot}), .xout(xbaserot_w), .yout(ybaserot_w), .phaseout(),.error(),.gin({active_delay,falingdelay}),.gout({activecordic_w,cordicdone_w}));
always @(posedge clk) begin
	xbaserot <= xbaserot_w;
	ybaserot <= ybaserot_w;
	activecordic <= activecordic_w;
	cordicdone <= cordicdone_w;
end
assign active=elactive|active_delay|activecordic;
assign resultx=resultx_r;//cordicdone ? xbaserot[31] : resultx;
assign resulty=resulty_r;//cordicdone ? ybaserot[31] : resulty;
assign xacc=xbaserot;//xaccr; buffer
assign yacc=ybaserot;//yaccr; buffer
assign done=cordicdone; //buffer
endmodule
