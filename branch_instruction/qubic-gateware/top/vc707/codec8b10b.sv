module test(ifcodec8b10b.port8b port8b);
endmodule
interface ifcodec8b10b(input txclk,input rxclk,input txreset,input rxreset);
	wire [7:0] tx8b;
	wire tx8bisk;
	reg txrd=0;
	wire [7:0] rx8b;
	wire rx8bisk;
	reg rxrd=0;
	reg rxdisperr=0;
	reg rxnointable=0;
	reg rxdispinit=0;

	wire [9:0] tx10b;
	wire [9:0] rx10b;

modport port8b(input rx8b,rx8bisk
,output tx8b,tx8bisk
);
modport port10b(input tx10b
,output rx10b
);

//reg [9:0] rx10br=0;
//always @(posedge rxclk) begin
//	rx10br<=rx10b;
//end

reg [4:0] RXEDCBA=0;
reg [2:0] RXHGF=0;
wire [5:0] rxabcdei;
wire [3:0] rxfghj;
assign {rxfghj[0],rxfghj[1],rxfghj[2],rxfghj[3],rxabcdei[0],rxabcdei[1],rxabcdei[2],rxabcdei[3],rxabcdei[4],rxabcdei[5]}=0;//rx10b;
assign rx8b ={RXHGF,RXEDCBA};
reg rxisk6=0;
reg rxisk4=0;
assign rx8bisk=rxisk6|rxisk4;
reg rxrd6=0;
reg rxrd4=0;
always @(*) begin
    case(rxabcdei)
        6'b100111:begin RXEDCBA=5'd00;rxisk6=1'b0;rxrd6=1'b1;end
        6'b011000:begin RXEDCBA=5'd00;rxisk6=1'b0;rxrd6=1'b1;end
        6'b011101:begin RXEDCBA=5'd01;rxisk6=1'b0;rxrd6=1'b1;end
        6'b100010:begin RXEDCBA=5'd01;rxisk6=1'b0;rxrd6=1'b1;end
        6'b101101:begin RXEDCBA=5'd02;rxisk6=1'b0;rxrd6=1'b1;end
        6'b010010:begin RXEDCBA=5'd02;rxisk6=1'b0;rxrd6=1'b1;end
        6'b110001:begin RXEDCBA=5'd03;rxisk6=1'b0;rxrd6=1'b0;end
        6'b110101:begin RXEDCBA=5'd04;rxisk6=1'b0;rxrd6=1'b1;end
        6'b001010:begin RXEDCBA=5'd04;rxisk6=1'b0;rxrd6=1'b1;end
        6'b101001:begin RXEDCBA=5'd05;rxisk6=1'b0;rxrd6=1'b0;end
        6'b011001:begin RXEDCBA=5'd06;rxisk6=1'b0;rxrd6=1'b0;end
        6'b111000:begin RXEDCBA=5'd07;rxisk6=1'b0;rxrd6=1'b1;end
        6'b000111:begin RXEDCBA=5'd07;rxisk6=1'b0;rxrd6=1'b1;end
        6'b111001:begin RXEDCBA=5'd08;rxisk6=1'b0;rxrd6=1'b1;end
        6'b000110:begin RXEDCBA=5'd08;rxisk6=1'b0;rxrd6=1'b1;end
        6'b100101:begin RXEDCBA=5'd09;rxisk6=1'b0;rxrd6=1'b0;end
        6'b010101:begin RXEDCBA=5'd10;rxisk6=1'b0;rxrd6=1'b0;end
        6'b110100:begin RXEDCBA=5'd11;rxisk6=1'b0;rxrd6=1'b0;end
        6'b001101:begin RXEDCBA=5'd12;rxisk6=1'b0;rxrd6=1'b0;end
        6'b101100:begin RXEDCBA=5'd13;rxisk6=1'b0;rxrd6=1'b0;end
        6'b011100:begin RXEDCBA=5'd14;rxisk6=1'b0;rxrd6=1'b0;end
        6'b010111:begin RXEDCBA=5'd15;rxisk6=1'b0;rxrd6=1'b1;end
        6'b101000:begin RXEDCBA=5'd15;rxisk6=1'b0;rxrd6=1'b1;end
        6'b011011:begin RXEDCBA=5'd16;rxisk6=1'b0;rxrd6=1'b1;end
        6'b100100:begin RXEDCBA=5'd16;rxisk6=1'b0;rxrd6=1'b1;end
        6'b100011:begin RXEDCBA=5'd17;rxisk6=1'b0;rxrd6=1'b0;end
        6'b010011:begin RXEDCBA=5'd18;rxisk6=1'b0;rxrd6=1'b0;end
        6'b110010:begin RXEDCBA=5'd19;rxisk6=1'b0;rxrd6=1'b0;end
        6'b001011:begin RXEDCBA=5'd20;rxisk6=1'b0;rxrd6=1'b0;end
        6'b101010:begin RXEDCBA=5'd21;rxisk6=1'b0;rxrd6=1'b0;end
        6'b011010:begin RXEDCBA=5'd22;rxisk6=1'b0;rxrd6=1'b0;end
        6'b111010:begin RXEDCBA=5'd23;rxisk6=1'b0;rxrd6=1'b1;end
        6'b000101:begin RXEDCBA=5'd23;rxisk6=1'b0;rxrd6=1'b1;end
        6'b110011:begin RXEDCBA=5'd24;rxisk6=1'b0;rxrd6=1'b1;end
        6'b001100:begin RXEDCBA=5'd24;rxisk6=1'b0;rxrd6=1'b1;end
        6'b100110:begin RXEDCBA=5'd25;rxisk6=1'b0;rxrd6=1'b0;end
        6'b010110:begin RXEDCBA=5'd26;rxisk6=1'b0;rxrd6=1'b0;end
        6'b110110:begin RXEDCBA=5'd27;rxisk6=1'b0;rxrd6=1'b1;end
        6'b001001:begin RXEDCBA=5'd27;rxisk6=1'b0;rxrd6=1'b1;end
        6'b001110:begin RXEDCBA=5'd28;rxisk6=1'b0;rxrd6=1'b0;end
        6'b001111:begin RXEDCBA=5'd28;rxisk6=1'b1;rxrd6=1'b1;end
        6'b110000:begin RXEDCBA=5'd28;rxisk6=1'b1;rxrd6=1'b1;end
        6'b101110:begin RXEDCBA=5'd29;rxisk6=1'b0;rxrd6=1'b1;end
        6'b010001:begin RXEDCBA=5'd29;rxisk6=1'b0;rxrd6=1'b1;end
        6'b011110:begin RXEDCBA=5'd30;rxisk6=1'b0;rxrd6=1'b1;end
        6'b100001:begin RXEDCBA=5'd30;rxisk6=1'b0;rxrd6=1'b1;end
        6'b101011:begin RXEDCBA=5'd31;rxisk6=1'b0;rxrd6=1'b1;end
        6'b010100:begin RXEDCBA=5'd31;rxisk6=1'b0;rxrd6=1'b1;end
        default:begin RXEDCBA=5'd00;rxisk6=1'b0;rxrd6=1'b0;end
    endcase
end


always @(*) begin
    case(rxfghj)
        4'b0001:begin RXHGF=3'd7;rxisk4=1'b0;rxrd4=1'b1;end
        4'b0010:begin RXHGF=3'd4;rxisk4=rxisk6;rxrd4=1'b1;end
        4'b0011:begin RXHGF=3'd3;rxisk4=rxisk6;rxrd4=1'b0;end
        4'b0100:begin RXHGF=3'd0;rxisk4=rxisk6;rxrd4=1'b1;end
        4'b0101:begin RXHGF=(rxisk6&~rxrd) ? 3'd5 : 3'd2;rxisk4=rxisk6;rxrd4=1'b0;end
        4'b0110:begin RXHGF=(rxisk6&~rxrd) ? 3'd1 : 3'd6;rxisk4=rxisk6;rxrd4=1'b0;end
        4'b0111:begin RXHGF=3'd7;rxisk4=~|{(RXEDCBA==5'd17),(RXEDCBA==5'd18),(RXEDCBA==5'd20)}|rxisk6;rxrd4=1'b1;end
        4'b1000:begin RXHGF=3'd7;rxisk4=~|{(RXEDCBA==5'd11),(RXEDCBA==5'd13),(RXEDCBA==5'd14)}|rxisk6;rxrd4=1'b1;end
        4'b1001:begin RXHGF=(rxisk6&~rxrd) ? 3'd6 : 3'd1;rxisk4=rxisk6;rxrd4=1'b0;end
        4'b1010:begin RXHGF=(rxisk6&~rxrd) ? 3'd2 : 3'd5;rxisk4=rxisk6;rxrd4=1'b0;end
        4'b1011:begin RXHGF=3'd0;rxisk4=rxisk6;rxrd4=1'b1;end
        4'b1100:begin RXHGF=3'd3;rxisk4=rxisk6;rxrd4=1'b1;end
        4'b1101:begin RXHGF=3'd4;rxisk4=rxisk6;rxrd4=1'b1;end
        4'b1110:begin RXHGF=3'd7;rxisk4=1'b0;rxrd4=1'b1;end
    endcase
end
localparam K285P=10'b1100000101;
localparam K285M=10'b0011111010;
wire rxrdcalc=rxrd6 ? ~(rxrd^rxrd4) : (rxrd^rxrd4);
wire rxrdset;//=rx10b==K285P ? 1'b1 : rx10b==K285M ? 1'b0 : 1'b0;// rxrdcalc;
reg rxdisperr_r=0;
always @(posedge rxclk) begin
//	rxdispinit<= rxreset ? 1'b0 : rx10b==K285P ? 1'b1 : rx10b==K285M ? 1'b1 : rxdispinit;
//	rxrd<=rxreset ? 0 : rx10b==K285P ? 1'b1 : rx10b==K285M ? 1'b0 :  rxrdcalc;
//	rxdisperr<= rxreset ? 0 : ~rxdispinit ? 0 : (rx10b==K285P & ~rxrdcalc) | (rx10b==K285M & rxrdcalc);
	rxdisperr_r<= rxreset ? 0 : rxdisperr ? 1'b1 : rxdisperr_r;
end

wire [4:0] TXEDCBA;
wire [2:0] TXHGF;
assign {TXHGF,TXEDCBA}=tx8b;
reg [5:0] txabcdei=0;
reg [3:0] txfghj=0;
reg txrd6=0;
reg txrd4=0;
always @(*) begin
    case(TXEDCBA)
        5'b00000:begin txabcdei=~txrd?6'b100111:6'b011000;txrd6=1'b1;end
        5'b00001:begin txabcdei=~txrd?6'b011101:6'b100010;txrd6=1'b1;end
        5'b00010:begin txabcdei=~txrd?6'b101101:6'b010010;txrd6=1'b1;end
        5'b00011:begin txabcdei=6'b110001;txrd6=1'b0;end
        5'b00100:begin txabcdei=~txrd?6'b110101:6'b001010;txrd6=1'b1;end
        5'b00101:begin txabcdei=6'b101001;txrd6=1'b0;end
        5'b00110:begin txabcdei=6'b011001;txrd6=1'b0;end
        5'b00111:begin txabcdei=~txrd?6'b111000:6'b000111;txrd6=1'b0;end
        5'b01000:begin txabcdei=~txrd?6'b111001:6'b000110;txrd6=1'b1;end
        5'b01001:begin txabcdei=6'b100101;txrd6=1'b0;end
        5'b01010:begin txabcdei=6'b010101;txrd6=1'b0;end
        5'b01011:begin txabcdei=6'b110100;txrd6=1'b0;end
        5'b01100:begin txabcdei=6'b001101;txrd6=1'b0;end
        5'b01101:begin txabcdei=6'b101100;txrd6=1'b0;end
        5'b01110:begin txabcdei=6'b011100;txrd6=1'b0;end
        5'b01111:begin txabcdei=~txrd?6'b010111:6'b101000;txrd6=1'b1;end
        5'b10000:begin txabcdei=~txrd?6'b011011:6'b100100;txrd6=1'b1;end
        5'b10001:begin txabcdei=6'b100011;txrd6=1'b0;end
        5'b10010:begin txabcdei=6'b010011;txrd6=1'b0;end
        5'b10011:begin txabcdei=6'b110010;txrd6=1'b0;end
        5'b10100:begin txabcdei=6'b001011;txrd6=1'b0;end
        5'b10101:begin txabcdei=6'b101010;txrd6=1'b0;end
        5'b10110:begin txabcdei=6'b011010;txrd6=1'b0;end
        5'b10111:begin txabcdei=~txrd?6'b111010:6'b000101;txrd6=1'b1;end
        5'b11000:begin txabcdei=~txrd?6'b110011:6'b001100;txrd6=1'b1;end
        5'b11001:begin txabcdei=6'b100110;txrd6=1'b0;end
        5'b11010:begin txabcdei=6'b010110;txrd6=1'b0;end
        5'b11011:begin txabcdei=~txrd?6'b110110:6'b001001;txrd6=1'b1;end
        5'b11100:begin txabcdei=~tx8bisk? 6'b001110:~txrd?6'b001111:6'b110000;txrd6= ~tx8bisk ? 1'b0 : 1'b1;end
        5'b11101:begin txabcdei=~txrd?6'b101110:6'b010001;txrd6=1'b1;end
        5'b11110:begin txabcdei=~txrd?6'b011110:6'b100001;txrd6=1'b1;end
        5'b11111:begin txabcdei=~txrd?6'b101011:6'b010100;txrd6=1'b1;end
        //not used:begin txabcdei=~txrd?6'b111100:6'b000011;txrd6=1'b0;end
        default:begin txabcdei=6'b000000;txrd6=1'b0;end
    endcase
end


always @(*) begin
    if (~tx8bisk) begin
        case (TXHGF)
            3'b000:begin txfghj=~(txrd^txrd6)?4'b1011:4'b0100;txrd4=1'b1; end
            3'b001:begin txfghj=4'b1001;txrd4=1'b0; end
            3'b010:begin txfghj=4'b0101;txrd4=1'b0; end
            3'b011:begin txfghj=~(txrd^txrd6)?4'b1100:4'b0011;txrd4=1'b0; end
            3'b100:begin txfghj=~(txrd^txrd6)?4'b1101:4'b0010;txrd4=1'b1; end
            3'b101:begin txfghj=4'b1010;txrd4=1'b0; end
            3'b110:begin txfghj=4'b0110;txrd4=1'b0; end
            3'b111:begin txfghj= ~(txrd^txrd6)? (((TXEDCBA==5'd17)|(TXEDCBA==5'd18)|(TXEDCBA==5'd20)) ? 4'b0111:4'b1110): 
                (((TXEDCBA==5'd11)|(TXEDCBA==5'd13)|(TXEDCBA==5'd14)) ? 4'b1000:4'b0001); txrd4=1'b1; end
        endcase
    end
    else begin
        case (TXHGF)
            3'b000:begin txfghj=~(txrd^txrd6)?4'b1011:4'b0100;txrd4=1'b1; end
            3'b001:begin txfghj=~(txrd^txrd6)?4'b0110:4'b1001;txrd4=1'b0; end
            3'b010:begin txfghj=~(txrd^txrd6)?4'b1010:4'b0101;txrd4=1'b0; end
            3'b011:begin txfghj=~(txrd^txrd6)?4'b1100:4'b0011;txrd4=1'b0; end
            3'b100:begin txfghj=~(txrd^txrd6)?4'b1101:4'b0010;txrd4=1'b1; end
            3'b101:begin txfghj=~(txrd^txrd6)?4'b0101:4'b1010;txrd4=1'b0; end
            3'b110:begin txfghj=~(txrd^txrd6)?4'b1001:4'b0110;txrd4=1'b0; end
            3'b111:begin txfghj=~(txrd^txrd6)?4'b0111:4'b1000;txrd4=1'b1; end
        endcase
    end
end
assign tx10b={txfghj[0],txfghj[1],txfghj[2],txfghj[3],txabcdei[0],txabcdei[1],txabcdei[2],txabcdei[3],txabcdei[4],txabcdei[5]};
//assign tx10b={txabcdei,txfghj};//[0],txfghj[1],txfghj[2],txfghj[3],txabcdei[0],txabcdei[1],txabcdei[2],txabcdei[3],txabcdei[4],txabcdei[5]};
always @(posedge txclk) begin
	txrd<=txreset ? 0 : txrd4 ? ~(txrd^txrd6) : (txrd^txrd6);
end

endinterface
