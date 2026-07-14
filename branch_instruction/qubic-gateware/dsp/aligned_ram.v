/**
* RAM module where read_width = N*write_width; i.e. 
* the output word has a width/addressing scheme that is 
* some power-2 integer multiple of the input width/addressing scheme.
*
* 4x32 example below
*
* w_addr |   w_data
*   0    | data0[31:0] ---\
*   1    | data1[31:0]     \_____ read_data[0]
*   2    | data2[31:0]     /
*   3    | data3[31:0] ---/
*/

module aligned_ram #(
    parameter DIN_WIDTH=32,
    parameter N_DIN_TO_DOUT=4,
    parameter DOUT_ADDR_WIDTH=10,
    parameter READ_LATENCY=2)(
    input clk,
    input[DIN_WIDTH-1:0] write_data,
    input[DOUT_ADDR_WIDTH + $clog2(N_DIN_TO_DOUT)-1:0] write_addr,
    input write_enable,
    input[DOUT_ADDR_WIDTH-1:0] read_addr,
    output [N_DIN_TO_DOUT*DIN_WIDTH-1:0] read_data);

    localparam LOG_DIN_TO_DOUT = $clog2(N_DIN_TO_DOUT);

    reg[DIN_WIDTH-1:0] data[N_DIN_TO_DOUT*2**DOUT_ADDR_WIDTH-1:0];
    reg[DOUT_ADDR_WIDTH-1:0] cur_read_addr[READ_LATENCY-1:0];

    always @(posedge clk) begin
        cur_read_addr[0] <= read_addr;
        if(write_enable)
            data[write_addr] <= write_data;
    end

    genvar i;
    generate
        for(i=0; i<N_DIN_TO_DOUT; i=i+1) begin
            assign read_data[DIN_WIDTH*(i+1) - 1 : DIN_WIDTH*i] = data[(cur_read_addr[READ_LATENCY-1] * N_DIN_TO_DOUT) + i];
        end
    endgenerate

    generate for(i=1; i<READ_LATENCY; i=i+1) begin
        always @(posedge clk)
            cur_read_addr[i] <= cur_read_addr[i-1];
    end
    endgenerate

endmodule
    
