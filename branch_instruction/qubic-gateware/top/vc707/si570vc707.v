module si570vc707(input clk
,input [2:0] hs_div
,input [6:0] n1
,input [37:0] rfreq
,input start
,input smallchange
,output busy
// for small change, set smallchange=1, n1=0 and hs_div=0
,output [36:0] i2ccmd
,output i2cstart
,input i2cbusy
,input [2:0] hs_div_now
,input [6:0] n1_now
,input [37:0] rfreq_now
,input [5:0] newnow
,output [37:0] dbrfreq_w
,output [37:0] dbsmallmax
,output [37:0] dbsmallmin
,output [5:0] dbnewnow
,output [3:0] dbstate
,output [3:0] dbnext
);

reg [2:0] hs_div_new=0;
reg [6:0] n1_new=0;
reg [2:0] hs_div_r=0;
reg [6:0] n1_r=0;
reg [37:0] rfreq_r=0;
reg [37:0] rfreq_new=0;
reg [37:0] rfreq_w=0;
localparam RFREQINIT=38'h2bc018e2a;
reg [37:0] smallmax=RFREQINIT+(RFREQINIT>>9);
reg [37:0] smallmin=RFREQINIT-(RFREQINIT>>9);
assign dbrfreq_w=rfreq_w;
assign dbsmallmax=smallmax;
assign dbsmallmin=smallmin;
reg busy_r=0;
reg [36:0] i2ccmd_r=0;
reg i2cstart_r=0;
wire signed [38:0] deltarfreq;
assign 	deltarfreq=$signed(rfreq_r)-$signed(rfreq_now);
reg start_r=0;
reg smallchange_r=0;
assign dbnewnow=newnow;
always @(posedge clk) begin
	start_r<=start;
	if (start) begin
		rfreq_r<=rfreq;
		n1_r<=n1;
		hs_div_r<=hs_div;
		smallchange_r<=smallchange;
	end
	//smallmax<=rfreq_now+(rfreq_now>>9);
	//smallmin<=rfreq_now-(rfreq_now>>9);
end
wire smallppm= &deltarfreq[38:29] | (~|deltarfreq[38:29]);
wire midstep=smallchange_r & ~smallppm  & (&newnow) ;
reg midstep_r=0;

localparam IDLE=4'h0;
localparam START=4'h1;
localparam START2=4'h2;
localparam I2CSW=4'h3;
localparam SMALLFRZ=4'h4;
localparam LARGEFRZ=4'h5;
localparam REG7=4'h6;
localparam REG8=4'h7;
localparam REG9=4'h8;
localparam REGA=4'h9;
localparam REGB=4'ha;
localparam REGC=4'hb;
localparam SMALLUNFRZ=4'hc;
localparam LARGEUNFRZ=4'hd;
localparam NEWFREQ=4'he;
//localparam REG70=4'hf;
localparam CNT=16'd5;
reg [15:0] cnt=0;
reg [3:0] state=IDLE;
reg [3:0] next=IDLE;
assign dbstate=state;
assign dbnext=next;
reg busy1=0;
always @(posedge clk) begin
	state<=next;
	cnt<= (state==next)&(state!=IDLE) ? (&cnt ? cnt : cnt+1) : 0;
end
always @(*) begin
	case (state)
		IDLE:begin next = start_r ? START: IDLE; end
		START:begin next = ~i2cbusy ? I2CSW : START; end
		I2CSW:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? START2 :I2CSW; end
		START2:begin next = ~i2cbusy ? smallchange_r ? SMALLFRZ : LARGEFRZ : START2; end
		SMALLFRZ:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? REG8 : SMALLFRZ ; end
		LARGEFRZ:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? REG7 : LARGEFRZ ; end
		REG7:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? REG8 : REG7; end
//		REG70:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? REG7 : REG70; end
		REG8:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? REG9 : REG8; end
		REG9:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? REGA : REG9; end
		REGA:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? REGB : REGA; end
		REGB:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? REGC : REGB; end
		REGC:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? smallchange_r ? SMALLUNFRZ : LARGEUNFRZ : REGC ; end
		SMALLUNFRZ:begin next = busy1 &(cnt>CNT) & ~i2cbusy ? (midstep_r) ?  I2CSW : IDLE : SMALLUNFRZ; end
		LARGEUNFRZ:begin next = busy1 &(cnt>CNT) & ~i2cbusy ?  NEWFREQ : LARGEUNFRZ ; end
		NEWFREQ:begin next = busy1 &(cnt>CNT) & ~i2cbusy ?  IDLE : NEWFREQ; end
		default: next = IDLE;
	endcase
end
always @(posedge clk) begin
	if (state==next)
		if (i2cbusy)
			busy1<=1'b1;
	else
		busy1<=1'b0;
	case (next)
		IDLE:begin
			busy_r<=1'b0;
			i2cstart_r<=0;
			i2ccmd_r<=0;
		end
		START:begin
			rfreq_new<=rfreq_r;
			n1_new<=n1_r;
			hs_div_new<=hs_div_r;
			busy_r<=1'b1;
			i2cstart_r<=0;
			i2ccmd_r<=0;
		end
		I2CSW:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h2,7'h74,1'h0,8'h1,16'h0};
		end
		START2:begin
			rfreq_w<= midstep ? (deltarfreq[38] ? smallmin : smallmax) : rfreq_r;
			midstep_r<=midstep&1'b0;
			i2cstart_r<=0;
			i2ccmd_r<=0;
		end
		SMALLFRZ:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'd135,8'h20,8'h0};
		end
		LARGEFRZ:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'd137,8'h10,8'h0};
		end
/*		REG70:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'h7,hs_div_new[2:0],n1_new[6:2],8'h0};
		end
		*/
		REG7:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'h7,hs_div_new[2:0],n1_new[6:2],8'h0};
		end
		REG8:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'h8,n1_new[1:0],rfreq_w[37:32],8'h0};
		end
		REG9:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'h9,rfreq_w[31:24],8'h0};
		end
		REGA:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'hA,rfreq_w[23:16],8'h0};
		end
		REGB:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'hB,rfreq_w[15:8],8'h0};
		end
		REGC:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'hC,rfreq_w[7:0],8'h0};
		end
		SMALLUNFRZ:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'd135,8'h00,8'h0};
		end
		LARGEUNFRZ:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'd137,8'h00,8'h0};
		end
		NEWFREQ:begin
			i2cstart_r<=~|cnt;
			i2ccmd_r<={1'b1,4'h3,7'h5d,1'h0,8'd135,8'h40,8'h0};
		end
	endcase
end

assign i2ccmd=i2ccmd_r;
assign i2cstart=i2cstart_r;
assign busy=busy_r;
endmodule
