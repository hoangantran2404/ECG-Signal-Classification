`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 02/09/2026 09:39:16 PM
// Design Name: 
// Module Name: ALU
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


module ALU #(
    parameter DATA_WIDTH=16,
    parameter ADDR_WIDTH=8
)
(
    input wire                          clk,
    input wire                          rst_n,
    input wire                          start_in,
    input wire signed [DATA_WIDTH-1:0]  weight_in,
    input wire signed [DATA_WIDTH-1:0]  activation_in,

    input wire signed [DATA_WIDTH-1:0]  bias_in,
    input wire                          bias_valid_in,
    input wire                          ReLU_en_in,
    input wire                          CTRL_send_enable_in,   
    input wire                          CTRL_gap_enable_in,


    output wire signed [DATA_WIDTH-1:0] ALU_data_out,
    output wire                         ALU_data_valid_out
);
    wire signed [DATA_WIDTH-1:0] MAC_out_wr;
    wire                         MAC_valid_wr;
    
    assign ALU_data_out         = (CTRL_send_enable_in && MAC_valid_wr)? MAC_out_wr : 0;
    assign ALU_data_valid_out   = (CTRL_send_enable_in && MAC_valid_wr) ? 1 : 0;

    MAC #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) MAC_inst (
        .clk                    (clk            ),
        .rst_n                  (rst_n          ),
        .CTRL_gap_enable_in     (CTRL_gap_enable_in),
        .ReLU_en_in             (ReLU_en_in     ),
        .weight_in              (weight_in      ),
        .activation_in          (activation_in  ),
        .bias_in                (bias_in        ),
        .bias_valid_in          (bias_valid_in  ),
        .MAC_out                (MAC_out_wr     ),
        .MAC_valid_out          (MAC_valid_wr   )
    );
endmodule

module MAC #(
    parameter DATA_WIDTH=16,
    parameter ADDR_WIDTH=8
)
(
    input wire clk,
    input wire rst_n,
    input wire                         CTRL_gap_enable_in,
    input wire                         ReLU_en_in,
    input wire signed [DATA_WIDTH-1:0] weight_in,
    input wire signed [DATA_WIDTH-1:0] activation_in,

    input wire signed [DATA_WIDTH-1:0] bias_in,
    input wire                         bias_valid_in,

    output reg signed [DATA_WIDTH-1:0] MAC_out,
    output reg                         MAC_valid_out
);  
    wire signed [2*DATA_WIDTH-1:0] partial_sum0_wr, partial_sum1_wr, partial_sum2_wr, partial_sum3_wr;
    wire signed[2*DATA_WIDTH-1:0] partial_sum4_wr, partial_sum5_wr, partial_sum6_wr, partial_sum7_wr;
    wire signed [2*DATA_WIDTH-1:0] partial_sum8_wr, partial_sum9_wr,partial_sum10_wr, partial_sum11_wr; 
    wire signed[2*DATA_WIDTH-1:0] partial_sum12_wr, partial_sum13_wr, partial_sum14_wr, partial_sum15_wr;
    wire signed [2*DATA_WIDTH-1:0] sum_stage1_wr, sum_stage2_wr, sum_final_wr;

    wire signed [DATA_WIDTH-1:0] multiplicand_wr;
    wire signed [DATA_WIDTH-1:0] multiplier_wr;
    wire                         sign_wr;
    wire signed [DATA_WIDTH-1:0] bias_wr;
    wire signed [DATA_WIDTH-1:0] out_wr;
    wire signed [DATA_WIDTH-1:0] out1_wr;

    reg signed [DATA_WIDTH-1:0] sum_stage1_rg;
    reg signed [DATA_WIDTH-1:0] accumulation_rg;
    reg signed [DATA_WIDTH-1:0] multiplier_rg;
    reg signed [DATA_WIDTH-1:0] multiplicand_rg;
    reg signed [DATA_WIDTH-1:0] bias_rg;
    reg                         sign_rg;
    reg                         ReLU_en_rg;
    reg                         bias_valid_rg;
//Clock1
    
    assign multiplicand_wr  =       (CTRL_gap_enable_in)? 16'd64 :
                                    (weight_in<0) ? (-weight_in) : weight_in;                                              // Absolute value of the weight
    assign multiplier_wr    =       (activation_in<0) ? (-activation_in) : activation_in;                                    // Absolute value of the activation
    assign sign_wr          =       (CTRL_gap_enable_in)? activation_in[DATA_WIDTH-1] :
                                    (weight_in[DATA_WIDTH-1] ^ activation_in[DATA_WIDTH-1]);                          // Determine the sign of the product                                     // Sign-extend the weight to 32 bits

    assign partial_sum0_wr = multiplier_wr[0] ? ($signed(multiplicand_wr) << 0) : 0;
    assign partial_sum1_wr = multiplier_wr[1] ? ($signed(multiplicand_wr) << 1) : 0;
    assign partial_sum2_wr = multiplier_wr[2] ? ($signed(multiplicand_wr) << 2) : 0;
    assign partial_sum3_wr = multiplier_wr[3] ? ($signed(multiplicand_wr) << 3) : 0;
    assign partial_sum4_wr = multiplier_wr[4] ? ($signed(multiplicand_wr) << 4) : 0;
    assign partial_sum5_wr = multiplier_wr[5] ? ($signed(multiplicand_wr) << 5) : 0;
    assign partial_sum6_wr = multiplier_wr[6] ? ($signed(multiplicand_wr) << 6) : 0;
    assign partial_sum7_wr = multiplier_wr[7] ? ($signed(multiplicand_wr) << 7) : 0;
    assign partial_sum8_wr = multiplier_wr[8] ? ($signed(multiplicand_wr) << 8) : 0;
    assign partial_sum9_wr = multiplier_wr[9] ? ($signed(multiplicand_wr) << 9) : 0;
    
    assign sum_stage1_wr = partial_sum0_wr + partial_sum1_wr + partial_sum2_wr + partial_sum3_wr +
                          partial_sum4_wr + partial_sum5_wr + partial_sum6_wr + partial_sum7_wr + partial_sum8_wr + partial_sum9_wr;
//Clock2    
    assign partial_sum10_wr = multiplier_rg[10] ? ($signed(multiplicand_rg) << 10) : 0;
    assign partial_sum11_wr = multiplier_rg[11] ? ($signed(multiplicand_rg) << 11) : 0;
    assign partial_sum12_wr = multiplier_rg[12] ? ($signed(multiplicand_rg) << 12) : 0;
    assign partial_sum13_wr = multiplier_rg[13] ? ($signed(multiplicand_rg) << 13) : 0;
    assign partial_sum14_wr = multiplier_rg[14] ? ($signed(multiplicand_rg) << 14) : 0;
    assign partial_sum15_wr = multiplier_rg[15] ? ($signed(multiplicand_rg) << 15) : 0;
    assign sum_stage2_wr    = partial_sum10_wr + partial_sum11_wr + partial_sum12_wr + partial_sum13_wr 
                            + partial_sum14_wr + partial_sum15_wr + sum_stage1_rg;

    assign sum_final_wr = (sign_rg)? $signed(sum_stage2_wr[21:6]+sum_stage2_wr[5:5]): 
                                    -$signed(sum_stage2_wr[21:6]+sum_stage2_wr[5:5]); // Convert to Q9.6 by shift right or divide by 64 and sign correction
    assign bias_wr = (bias_valid_rg) ? bias_rg : 0;
    assign out_wr  = sum_final_wr + bias_wr + accumulation_rg;

    assign out1_wr = ((ReLU_en_rg) && (out_wr[DATA_WIDTH-1:DATA_WIDTH-1]==1)) ? 0 : out_wr; // ReLU activation function

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulation_rg <= 0;
            multiplier_rg   <= 0;
            multiplicand_rg <= 0;
            sum_stage1_rg   <= 0; // Reset cho Reg Pipeline
            bias_rg         <= 0;
            sign_rg         <= 0;
            ReLU_en_rg      <= 0;
            bias_valid_rg   <= 0;
            MAC_out         <= 0;
            MAC_valid_out   <= 0;
        end else begin
            sum_stage1_rg           <= sum_stage1_wr;
            multiplicand_rg         <= multiplicand_wr;
            multiplier_rg           <= multiplier_wr;
            sign_rg                 <= sign_wr;

            ReLU_en_rg              <= ReLU_en_in;
            bias_rg                 <= bias_in;
            bias_valid_rg           <= bias_valid_in;
            if (bias_valid_rg) begin 
                MAC_out         <= out1_wr;
                MAC_valid_out   <= 1'b1;
                accumulation_rg <= 0;
            end else begin
                accumulation_rg <= out_wr;
                MAC_valid_out   <= 1'b0;
            end
        end
    end
endmodule
