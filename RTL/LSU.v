`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 02/09/2026 09:39:16 PM
// Design Name: 
// Module Name: LSU
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
module LSU #(
    parameter DATA_WIDTH=16,
    parameter ADDR_WIDTH=8
)(
    input wire clk,
    input wire rst_n,
    // From AXI
    input wire AXI_LDM_wea_in,
    input wire AXI_LDM_ena_in,
    input wire [ADDR_WIDTH-1:0] AXI_LDM_addra_in,
    input wire AXI_LDM_dina_in,
    input wire AXI_LDM_web_in,
    input wire AXI_LDM_enb_in,
    input wire [ADDR_WIDTH-1:0] AXI_LDM_addrb_in,
    input wire AXI_LDM_dinb_in,
    // From SGB
    input wire SGB_LDM_wea_in,
    input wire SGB_LDM_ena_in,
    input wire [ADDR_WIDTH-1:0] SGB_LDM_addra_in,
    input wire SGB_LDM_dina_in,
    //From CTRL
    input wire CTRL_LDM_wea_in,
    input wire CTRL_LDM_ena_in,
    input wire [ADDR_WIDTH-1:0] CTRL_LDM_addra_in,
    input wire CTRL_LDM_web_in,
    input wire CTRL_LDM_enb_in,
    input wire [ADDR_WIDTH-1:0] CTRL_LDM_addrb_in,

    //From ALU
    input wire ALU_web_in, ALU_enb_in,
    input wire signed [DATA_WIDTH-1:0] ALU_dout_in,
    input wire [ADDR_WIDTH-1:0] ALU_addr_in,

    output wire signed [DATA_WIDTH-1:0] LSU_dout0_out,
    output wire signed [DATA_WIDTH-1:0] LSU_dout1_out
);
    wire LDM_0_wea_wr,LDM_1_wea_wr;
    wire LDM_0_web_wr,LDM_1_web_wr;
    wire LDM_0_ena_wr,LDM_1_ena_wr;
    wire LDM_0_enb_wr,LDM_1_enb_wr;
    wire [ADDR_WIDTH-1:0] LDM_0_addra_wr, LDM_1_addra_wr;
    wire [ADDR_WIDTH-1:0] LDM_0_addrb_wr, LDM_1_addrb_wr;
    wire [DATA_WIDTH-1:0] LDM_0_dina_wr, LDM_1_dina_wr;
    wire [DATA_WIDTH-1:0] LDM_0_dinb_wr, LDM_1_dinb_wr;
    wire [ADDR_WIDTH-1:0] LDM_0_douta_wr, LDM_1_douta_wr;
    wire [ADDR_WIDTH-1:0] LDM_0_doutb_wr, LDM_1_doutb_wr;
//LDM0: Save Weights of channel.
    assign LDM_0_ena_wr = (AXI_LDM_ena_in && (AXI_LDM_addra_in[7:6]==0)) ? AXI_LDM_ena_in : 
                          (CTRL_LDM_ena_in && (CTRL_LDM_addra_in[7:6]==0)) ? CTRL_LDM_ena_in : 0;
    assign LDM_0_wea_wr = (AXI_LDM_wea_in && (AXI_LDM_addra_in[7:6]==0)) ? AXI_LDM_wea_in : 
                          (CTRL_LDM_wea_in && (CTRL_LDM_addra_in[7:6]==0)) ? CTRL_LDM_wea_in : 0;
    assign LDM_0_addra_wr = (AXI_LDM_ena_in && (AXI_LDM_addra_in[7:6]==0)) ? AXI_LDM_addra_in : 
                          (CTRL_LDM_ena_in && (CTRL_LDM_addra_in[7:6]==0)) ? CTRL_LDM_addra_in : 0;
    assign LDM_0_dina_wr = (AXI_LDM_ena_in && (AXI_LDM_addra_in[7:6]==0)) ? AXI_LDM_dina_in : 0;

    assign LDM_0_enb_wr = (AXI_LDM_enb_in && (AXI_LDM_addrb_in[7:6]==0)) ? AXI_LDM_enb_in : 
                          (CTRL_LDM_enb_in && (CTRL_LDM_addrb_in[7:6]==0)) ? CTRL_LDM_enb_in : 0;
    assign LDM_0_web_wr = (AXI_LDM_web_in && (AXI_LDM_addrb_in[7:6]==0)) ? AXI_LDM_web_in :
                          (CTRL_LDM_web_in && (CTRL_LDM_addrb_in[7:6]==0)) ? CTRL_LDM_web_in : 0;
    assign LDM_0_addrb_wr = (AXI_LDM_enb_in && (AXI_LDM_addrb_in[7:6]==0)) ? AXI_LDM_addrb_in :
                            (CTRL_LDM_enb_in && (CTRL_LDM_addrb_in[7:6]==0)) ? CTRL_LDM_addrb_in : 0;
    assign LDM_0_dinb_wr = (AXI_LDM_enb_in && (AXI_LDM_addrb_in[7:6]==0)) ? AXI_LDM_dinb_in : 0;
    
//LDM1: Save ALU_output
    assign LSU_dout0_out = LDM_0_douta_wr; 

    assign LDM_1_ena_wr = (CTRL_LDM_ena_in && (CTRL_LDM_addra_in[7:6]==2'b01)) ? CTRL_LDM_ena_in : 0;
    assign LDM_1_wea_wr = 1'b0; 
    assign LDM_1_addra_wr = (CTRL_LDM_ena_in && (CTRL_LDM_addra_in[7:6]==2'b01)) ? CTRL_LDM_addra_in : 0;
    assign LDM_1_dina_wr = 0;

    assign LSU_dout1_out = LDM_1_douta_wr; 
    assign LDM_1_enb_wr = ALU_enb_in;
    assign LDM_1_web_wr = ALU_web_in;
    assign LDM_1_addrb_wr = ALU_addr_in;
    assign LDM_1_dinb_wr = ALU_dout_in;

    BRAM2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) LDM0 (
        .clk(clk),
        .wea(LDM_0_wea_wr),
        .ena(LDM_0_ena_wr),
        .addra(LDM_0_addra_wr),
        .dina(LDM_0_dina_wr),
        .web(LDM_0_web_wr),
        .enb(LDM_0_enb_wr),
        .addrb(LDM_0_addrb_wr),
        .dinb(LDM_0_dinb_wr),
        .douta(LDM_0_douta_wr),
        .doutb(LDM_0_doutb_wr)
    );
    BRAM2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) LDM1 (
        .clk(clk),
        .wea(LDM_1_wea_wr),
        .ena(LDM_1_ena_wr),
        .addra(LDM_1_addra_wr),
        .dina(LDM_1_dina_wr),
        .web(LDM_1_web_wr),
        .enb(LDM_1_enb_wr),
        .addrb(LDM_1_addrb_wr),
        .dinb(LDM_1_dinb_wr),
        .douta(LDM_1_douta_wr),
        .doutb(LDM_1_doutb_wr)
    );

endmodule