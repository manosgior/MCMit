module alu
    #(parameter DATA_WIDTH=32)(
      input clk,
      input[7:0] ctrl,
      input[DATA_WIDTH-1:0] in0,
      input[DATA_WIDTH-1:0] in1,
      output reg[DATA_WIDTH-1:0] out);

    wire[DATA_WIDTH-1:0] id0, id1, add, sub, mask, masked_in;
    wire eq, le, ge, sub_oflow, majority;
    wire[$clog2(DATA_WIDTH)-1:0] num_ones, num_selected;

    reg[DATA_WIDTH-1:0] in0_reg, in1_reg, local_out;

    always @(posedge clk) begin
        in0_reg <= in0;
        in1_reg <= in1;
        out <= local_out;
    end

    assign id0 = in0_reg;
    assign id1 = in1_reg;
    assign add = in0_reg + in1_reg;
    assign sub = in0_reg - in1_reg;
    assign eq = (sub == 0);

    assign sub_oflow = (((~in0_reg[DATA_WIDTH-1]) & in1_reg[DATA_WIDTH-1] & sub[DATA_WIDTH-1])
                        | (in0_reg[DATA_WIDTH-1] & (~in1_reg[DATA_WIDTH-1]) & (~sub[DATA_WIDTH-1])));
    assign le = sub[DATA_WIDTH-1] ^ sub_oflow; //this assumes twos complement!
    assign ge = (~le) & (~eq);

    assign mask = in0_reg;
    assign masked_in = mask & in1_reg;
    assign num_ones = masked_in[0] + masked_in[1] + masked_in[2] + masked_in[3] + masked_in[4] +
                      masked_in[5] + masked_in[6] + masked_in[7] + masked_in[8] + masked_in[9] +
                      masked_in[10] + masked_in[11] + masked_in[12] + masked_in[13] + masked_in[14] +
                      masked_in[15] + masked_in[16] + masked_in[17] + masked_in[18] + masked_in[19] +
                      masked_in[20] + masked_in[21] + masked_in[22] + masked_in[23] + masked_in[24] +
                      masked_in[25] + masked_in[26] + masked_in[27] + masked_in[28] + masked_in[29] +
                      masked_in[30] + masked_in[31];
    assign num_selected = mask[0] + mask[1] + mask[2] + mask[3] + mask[4] + mask[5] + mask[6] +
                          mask[7] + mask[8] + mask[9] + mask[10] + mask[11] + mask[12] + mask[13] +
                          mask[14] + mask[15] + mask[16] + mask[17] + mask[18] + mask[19] +
                          mask[20] + mask[21] + mask[22] + mask[23] + mask[24] + mask[25] +
                          mask[26] + mask[27] + mask[28] + mask[29] + mask[30] + mask[31];
    assign majority = (num_selected == 0) ? 1'b0 :
                      (num_ones >= ((num_selected / 2) + 1)) ? 1'b1 : 1'b0;

    always @(*) begin
        case(ctrl)
            8'd0 : local_out = id0;
            8'd1 : local_out = add;
            8'd2 : local_out = sub;
            8'd3 : begin
                local_out[0] = eq;
                local_out[DATA_WIDTH-1:1] = 0;
            end
            8'd4 : begin
                local_out[0] = le;
                local_out[DATA_WIDTH-1:1] = 0;
            end
            8'd5 : begin
                local_out[0] = ge;
                local_out[DATA_WIDTH-1:1] = 0;
            end
            8'd6 : local_out = id1;
            8'd7 : local_out = 0;
            // Bitwise logic operations, in0 is the mask, in1 the data
            8'd8 : begin
                local_out[0] = (in0_reg & in1_reg) == in0_reg;
                local_out[DATA_WIDTH-1:1] = 0;
            end
            8'd9 : begin
                local_out[0] = |(in0_reg & in1_reg);
                local_out[DATA_WIDTH-1:1] = 0;
            end
            8'd10 : begin
                local_out[0] = ^(in0_reg & in1_reg);
                local_out[DATA_WIDTH-1:1] = 0;
            end
            8'd11 : begin
                local_out[0] = majority;
                local_out[DATA_WIDTH-1:1] = 0;
            end
            default : local_out = 0;
        endcase 
    end

endmodule
