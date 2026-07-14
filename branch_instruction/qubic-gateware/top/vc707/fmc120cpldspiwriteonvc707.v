module fmc120cpldspiwriteonvc707(
	input clk
	,input reset
	,input [31:0] devcmd
	,input stb_devcmd
	,input running
	,input [7:0] i2cswlocation
	,output busy
	,output start
	,output stopbit
	,output [3:0] nack
	,output [31:0] datatx
);
localparam FMC1CPLD=4'h0;
localparam FMC1LMK =4'h1;
localparam FMC1DAC =4'h2;
localparam FMC1ADCA=4'h3;
localparam FMC1ADCB=4'h4;
localparam FMC1LTC2657=4'h5;
localparam FMC2CPLD=4'h8;
localparam FMC2LMK =4'h9;
localparam FMC2DAC =4'ha;
localparam FMC2ADCA=4'hb;
localparam FMC2ADCB=4'hc;
localparam FMC2LTC2657=4'hd;


localparam SWITCHDEV=7'h74;
localparam SWITCHNACK=2;
localparam SWITCHSTOP=1;
wire [3:0] devsel_w;
wire [23:0] addrdata_w;
wire [2:0] zero3_w;
wire r1w0_w;
assign {devsel_w,zero3_w,r1w0_w,addrdata_w}=devcmd;
reg [3:0] devsel=0;
reg [23:0] addrdata=0;
reg [2:0] zero3=0;
reg r1w0=0; // should always be 0 for here now, read not down here yet
reg [3:0] nackval=0;
reg stopval=0;
reg [6:0] devaddr=0;
reg [7:0] switch=0;
reg [7:0] action=0;
always @(posedge clk) begin
	{devsel,zero3,r1w0,addrdata}<=devcmd;
	case (devsel_w)
		FMC1CPLD: begin	nackval<=3;stopval<=1; devaddr<=7'h1c; switch<=8'h2; action<=8'h0;	end
		FMC1LMK : begin	nackval<=3;stopval<=1; devaddr<=7'h1c; switch<=8'h2; action<=8'h1;	end
		FMC1DAC : begin	nackval<=3;stopval<=1; devaddr<=7'h1c; switch<=8'h2; action<=8'h2;	end
		FMC1ADCA: begin	nackval<=3;stopval<=1; devaddr<=7'h1c; switch<=8'h2; action<=8'h4;	end
		FMC1ADCB: begin	nackval<=3;stopval<=1; devaddr<=7'h1c; switch<=8'h2; action<=8'h8;	end
		FMC1LTC2657: begin	nackval<=3;stopval<=1; devaddr<=7'h10; switch<=8'h2; action<=8'h0;	end
		FMC2CPLD: begin	nackval<=3;stopval<=1; devaddr<=7'h1c; switch<=8'h4; action<=8'h0;	end
		FMC2LMK : begin	nackval<=3;stopval<=1; devaddr<=7'h1c; switch<=8'h4; action<=8'h1;	end
		FMC2DAC : begin	nackval<=3;stopval<=1; devaddr<=7'h1c; switch<=8'h4; action<=8'h2;	end
		FMC2ADCA: begin	nackval<=3;stopval<=1; devaddr<=7'h1c; switch<=8'h4; action<=8'h4;	end
		FMC2ADCB: begin	nackval<=3;stopval<=1; devaddr<=7'h1c; switch<=8'h4; action<=8'h8;	end
		FMC2LTC2657: begin	nackval<=3;stopval<=1; devaddr<=7'h10; switch<=8'h4;	end
	endcase
end



localparam IDLE=0;
//localparam PREP=
localparam SWITCH=1;
localparam SPI6=2;
localparam SPI7=3;
localparam SPI8=4;
localparam SPI0=5;
localparam CPLD=6;
localparam LTC2657=7;

reg busy_r=0;
reg start_r=0;
reg stopbit_r=0;
reg [3:0] nack_r=0;
reg [31:0] datatx_r=0;
reg [3:0] state=IDLE;
reg [3:0] next=IDLE;
reg [31:0] scnt=0;
reg [7:0] lastswitch=0;
wire scnt5=|scnt[31:3];
always @(posedge clk or posedge reset) begin
	if (reset) begin
		state<=IDLE;
	end
	else begin
		state<=next;
		scnt<=(state==next)&(state!=IDLE) ? scnt+1 : 0;
	end
end
always @(*) begin
	case (state)
		IDLE: next= stb_devcmd ? SWITCH : IDLE;
//		PREP: next=SWITCH;
		SWITCH: next = (scnt5& ~running)|(lastswitch==switch) ? ( ~|devsel[2:0] ? CPLD :(~devsel[2:0]==5) ? LTC2657 : SPI8) : SWITCH;
		CPLD: next=  scnt5 & ~running ? IDLE : CPLD ;
		SPI8: next= scnt5&  ~running ? SPI7 : SPI8 ;
		SPI7: next= scnt5& ~running ? SPI6 : SPI7 ;
		SPI6: next= scnt5& ~running ? SPI0 : SPI6 ;
		SPI0: next= scnt5& ~running ? IDLE : SPI0 ;
		LTC2657: next= scnt5& ~running ? IDLE : LTC2657 ;
	endcase
end
always @(posedge clk) begin
	if (reset) begin
		start_r<=1'b0;
		stopbit_r<=1'b0;
		nack_r<=1'b0;
		datatx_r<=1'b0;
		busy_r<=1'b0;
	end
	else begin
		case (next)
			IDLE: begin
				start_r<=1'b0;
				stopbit_r<=1'b0;
				nack_r<=1'b0;
				datatx_r<=1'b0;
				busy_r<=1'b0;
			end
			SWITCH: begin
				start_r<=scnt==2;
				stopbit_r<=SWITCHSTOP;
				nack_r<=SWITCHNACK;
				datatx_r<={SWITCHDEV,r1w0,switch,16'h0};
				busy_r<=1'b1;
				if (scnt==0)
					lastswitch<=i2cswlocation;
			end
			CPLD: begin
				start_r<=scnt==2;
				stopbit_r<=stopval;
				nack_r<=nackval;
				datatx_r<={devaddr,r1w0,addrdata};
				busy_r<=1'b1;
			end
			SPI8: begin
				start_r<=scnt==2;
				stopbit_r<=stopval;
				nack_r<=nackval;
				datatx_r<={devaddr,r1w0,8'h8,addrdata[23:16],8'h0};
				busy_r<=1'b1;
			end
			SPI7: begin
				start_r<=scnt==2;
				stopbit_r<=stopval;
				nack_r<=nackval;
				datatx_r<={devaddr,r1w0,8'h7,addrdata[15:8],8'h0};
				busy_r<=1'b1;
			end
			SPI6: begin
				start_r<=scnt==2;
				stopbit_r<=stopval;
				nack_r<=nackval;
				datatx_r<={devaddr,r1w0,8'h6,addrdata[7:0],8'h0};
				busy_r<=1'b1;
			end
			SPI0: begin
				start_r<=scnt==2;
				stopbit_r<=stopval;
				nack_r<=nackval;
				datatx_r<={devaddr,r1w0,8'h0,action,8'h0};
				busy_r<=1'b1;
			end
			LTC2657: begin
				start_r<=scnt==2;
				stopbit_r<=stopval;
				nack_r<=nackval;
				datatx_r<={devaddr,r1w0,addrdata};
				busy_r<=1'b1;
			end
			default: begin
				start_r<=1'b0;
				stopbit_r<=1'b0;
				nack_r<=1'b0;
				datatx_r<=1'b0;
				busy_r<=1'b0;
			end
		endcase
	end
end
assign start=start_r;
assign stopbit=stopbit_r;
assign nack=nack_r;
assign datatx=datatx_r;
assign busy=busy_r;

endmodule

/*
	devsel	fmc	i2csw	devaddr	action	r1w0	nack	stop	addrwidth	datawidth
switch				0x74		0	3	1
cpld	0x0	1	2	0x1c	0x00	0	3	1	8	8	16
lmk	0x1	1	2	4x1c	0x01	0	3	1	16	8	24
dac	0x2	1	2	3x1c	0x02	0	3	1	8	16	24
adca	0x3	1	2	1x1c	0x04	0	3	1	16	8	24
adcb	0x4	1	2	2x1c	0x08	0	3	1	16	8	24

ltc2657	0x5	1	2	0x10	0x00	0	3	1	8	16	24
cpld	0x8	2	4	0x1c	0x00	0	3	1	8	8	16
lmk	0x9	2	4	4x1c	0x01	0	3	1	16	8	24
dac	0xa	2	4	3x1c	0x02	0	3	1	8	16	24
adca	0xb	2	4	1x1c	0x04	0	3	1	16	8	24
adcb	0xc	2	4	2x1c	0x08	0	3	1	16	8	24

ltc2657	0xd	2	4	0x10	0x00	0	3	1	8	16	24
*/
