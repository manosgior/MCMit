//`ifndef instr_params_vh
//`define instr_params_vh

//ALU parameters
parameter ALU_ID0 = 8'b0000;
parameter ALU_ID1 = 8'b0110;
parameter ALU_ADD = 8'b0001;
parameter ALU_SUB = 8'b0010;
parameter ALU_EQ = 8'b0011;
parameter ALU_LE = 8'b0100;
parameter ALU_GE = 8'b0101;
parameter ALU_0 = 8'b0111;
parameter ALU_AND = 8'b1000;
parameter ALU_OR = 8'b1001;
parameter ALU_XOR = 8'b1010;
parameter ALU_MAJ = 8'b1011;

//in general: first 5 bits are opcode, followed by 3 bit ALU opcode

//5-bit opcode: 4-bit operation followed by LSB select for ALU_IN1 (0 for cmd, 1 for reg)
parameter PULSE_WRITE = 4'b1000;
parameter PULSE_WRITE_TRIG = 4'b1001;
parameter REG_ALU = 4'b0001; //|opcode[8]|cmd_value[32]|reg1_addr[4]|reg_write_addr[4]
parameter JUMP_I = 4'b0010; //|opcode[8]|cmd_value[32]|reg_addr[4]
parameter JUMP_COND = 4'b0011; //jump address is always immediate
parameter ALU_FPROC = 4'b0100;
parameter JUMP_FPROC = 4'b0101;
parameter JUMP_REDUCE_FPROC = 4'b1101;
parameter INC_QCLK = 4'b0110;
parameter SYNC = 4'b0111;
parameter DONE = 4'b1010;
parameter PULSE_RESET = 4'b1011;
parameter IDLE = 4'b1100;

//`endif
