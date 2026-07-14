module dsp_sim_toplevel#(
	`include "plps_para.vh"
	,`include "bram_para.vh")(
    input clk,
    input reset,
    input stb_start,
    input[31:0] nshot,
    input[31:0] mem_write_data,
    input[15:0] mem_write_addr,
    input[3:0] proc_write_sel, //index proc cores
    input[2:0] mem_write_sel, //0 for cmd, 1 for env, 2 for freq, etc
    input mem_write_en,
    input[12:0] buf_read_addr,
    input[ADC_AXIS_DATAWIDTH-1:0] adc[0:NADC-1],
    output[DAC_AXIS_DATAWIDTH-1:0] dac[0:NDAC-1],
    output[QUBIT_ACCBUF_W_DATAWIDTH-1:0] acc_read_data[0:NPROC-1],
    output[ACQBUF_W_DATAWIDTH-1:0] acq_read_data[0:NADC-1]);

    ifdsp dspif();

    //instantiate 3x qdrv elem mems
    genvar i;
    generate for(i=0; i<6; i=i+1) begin
        wire cmd_wen, env_qdrv_wen, freq_qdrv_wen, env_rdrv_wen, freq_rdrv_wen, env_rdlo_wen, freq_rdlo_wen;
        assign cmd_wen = (proc_write_sel == i) & (mem_write_sel == 0) & mem_write_en;
        aligned_ram #(.DIN_WIDTH(32), .N_DIN_TO_DOUT(5), .DOUT_ADDR_WIDTH(QUBIT_COMMAND_R_ADDRWIDTH), .READ_LATENCY(2))
            cmd_mem(.clk(clk), .write_data(mem_write_data), .write_addr(mem_write_addr[QUBIT_COMMAND_W_ADDRWIDTH-1:0]),
                .write_enable(cmd_wen), .read_addr(dspif.addr_qubit_command[i]), .read_data(dspif.data_qubit_command[i]));

        assign env_qdrv_wen = (proc_write_sel == i) & (mem_write_sel == 1) & mem_write_en;
        aligned_ram #(.DIN_WIDTH(32), .N_DIN_TO_DOUT(QUBIT_QDRV_ENV_R_DATAWIDTH/32), .DOUT_ADDR_WIDTH(QUBIT_QDRV_ENV_R_ADDRWIDTH), .READ_LATENCY(2))
            env_mem_qdrv(.clk(clk), .write_data(mem_write_data), .write_addr(mem_write_addr[QUBIT_QDRV_ENV_W_ADDRWIDTH-1:0]),
                .write_enable(env_qdrv_wen), .read_addr(dspif.addr_qubit_qdrv_env[i]), .read_data(dspif.data_qubit_qdrv_env[i]));
        assign freq_qdrv_wen = (proc_write_sel == i) & (mem_write_sel == 2) & mem_write_en;
        aligned_ram #(.DIN_WIDTH(32), .N_DIN_TO_DOUT(QUBIT_QDRV_FREQ_R_DATAWIDTH/32), .DOUT_ADDR_WIDTH(QUBIT_QDRV_FREQ_R_ADDRWIDTH), .READ_LATENCY(2))
            freq_mem_qdrv(.clk(clk), .write_data(mem_write_data), .write_addr(mem_write_addr[QUBIT_QDRV_FREQ_W_ADDRWIDTH-1:0]),
                .write_enable(freq_qdrv_wen), .read_addr(dspif.addr_qubit_qdrv_freq[i]), .read_data(dspif.data_qubit_qdrv_freq[i]));

        assign env_rdrv_wen = (proc_write_sel == i) & (mem_write_sel == 3) & mem_write_en;
        aligned_ram #(.DIN_WIDTH(32), .N_DIN_TO_DOUT(QUBIT_RDRV_ENV_R_DATAWIDTH/32), .DOUT_ADDR_WIDTH(QUBIT_RDRV_ENV_R_ADDRWIDTH), .READ_LATENCY(2))
            env_mem_rdrv(.clk(clk), .write_data(mem_write_data), .write_addr(mem_write_addr[QUBIT_RDRV_ENV_W_ADDRWIDTH-1:0]),
                .write_enable(env_rdrv_wen), .read_addr(dspif.addr_qubit_rdrv_env[i]), .read_data(dspif.data_qubit_rdrv_env[i]));
        assign freq_rdrv_wen = (proc_write_sel == i) & (mem_write_sel == 4) & mem_write_en;
        aligned_ram #(.DIN_WIDTH(32), .N_DIN_TO_DOUT(QUBIT_RDRV_FREQ_R_DATAWIDTH/32), .DOUT_ADDR_WIDTH(QUBIT_RDRV_FREQ_R_ADDRWIDTH), .READ_LATENCY(2))
            freq_mem_rdrv(.clk(clk), .write_data(mem_write_data), .write_addr(mem_write_addr[QUBIT_RDRV_FREQ_W_ADDRWIDTH-1:0]),
                .write_enable(freq_rdrv_wen), .read_addr(dspif.addr_qubit_rdrv_freq[i]), .read_data(dspif.data_qubit_rdrv_freq[i]));

        assign env_rdlo_wen = (proc_write_sel == i) & (mem_write_sel == 5) & mem_write_en;
        aligned_ram #(.DIN_WIDTH(32), .N_DIN_TO_DOUT(QUBIT_RDLO_ENV_R_DATAWIDTH/32), .DOUT_ADDR_WIDTH(QUBIT_RDLO_ENV_R_ADDRWIDTH), .READ_LATENCY(2))
            env_mem_rdlo(.clk(clk), .write_data(mem_write_data), .write_addr(mem_write_addr[QUBIT_RDLO_ENV_W_ADDRWIDTH-1:0]),
                .write_enable(env_rdlo_wen), .read_addr(dspif.addr_qubit_rdlo_env[i]), .read_data(dspif.data_qubit_rdlo_env[i]));
        assign freq_rdlo_wen = (proc_write_sel == i) & (mem_write_sel == 6) & mem_write_en;
        aligned_ram #(.DIN_WIDTH(32), .N_DIN_TO_DOUT(QUBIT_RDLO_FREQ_R_DATAWIDTH/32), .DOUT_ADDR_WIDTH(QUBIT_RDLO_FREQ_R_ADDRWIDTH), .READ_LATENCY(2))
            freq_mem_rdlo(.clk(clk), .write_data(mem_write_data), .write_addr(mem_write_addr[QUBIT_RDLO_FREQ_W_ADDRWIDTH-1:0]),
                .write_enable(freq_rdlo_wen), .read_addr(dspif.addr_qubit_rdlo_freq[i]), .read_data(dspif.data_qubit_rdlo_freq[i]));

        aligned_ram #(.DIN_WIDTH(QUBIT_ACCBUF_W_DATAWIDTH), .N_DIN_TO_DOUT(1), .DOUT_ADDR_WIDTH(QUBIT_ACCBUF_W_ADDRWIDTH), .READ_LATENCY(1))
            acc_buf(.clk(clk), .write_data(dspif.data_qubit_accbuf[i]), .write_addr(dspif.addr_qubit_accbuf[i]),
                .write_enable(dspif.we_qubit_accbuf[i]), .read_addr(buf_read_addr[QUBIT_ACCBUF_W_ADDRWIDTH-1:0]), .read_data(acc_read_data[i]));

    end
    endgenerate

    generate for(i=0; i<NADC; i=i+1)
        aligned_ram #(.DIN_WIDTH(ACQBUF_W_DATAWIDTH), .N_DIN_TO_DOUT(1), .DOUT_ADDR_WIDTH(ACQBUF_W_ADDRWIDTH), .READ_LATENCY(1))
            acq_buf(.clk(clk), .write_data(dspif.data_acqbuf[i]), .write_addr(dspif.addr_acqbuf[i]),
                .write_enable(dspif.we_acqbuf[i]), .read_addr(buf_read_addr[ACQBUF_W_ADDRWIDTH-1:0]), .read_data(acq_read_data[i]));
    endgenerate



    assign dac = dspif.dac;
    assign dspif.adc = adc;
    assign dspif.clk = clk;
    assign dspif.reset = reset;
    assign dspif.resetacc = reset;
    assign dspif.stb_start = stb_start;
    assign dspif.nshot = nshot;
    assign dspif.acc_shift = 15;

    generate for(i=0; i<3; i=i+1) begin
        assign dspif.coef[i][i] = 32'h7fff0000;
    end
    endgenerate

    dsp dspmod(.dspif(dspif));

endmodule



