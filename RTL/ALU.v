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
    parameter DATA_WIDTH=16
)
(
    input wire                          clk,
    input wire                          rst_n,
    input wire signed [DATA_WIDTH-1:0]  Filter_in,
    input wire signed [DATA_WIDTH-1:0]  IA_in,
    input wire [2:0]                    CTRL_counter_in,



    output wire signed [DATA_WIDTH-1:0] ALU_data_out,
    output wire                         ALU_data_valid_out
);
    wire signed [DATA_WIDTH-1:0]        MAC_out_wr;
    wire                                MAC_valid_wr;
    
    assign ALU_data_out         = (MAC_valid_wr)? MAC_out_wr : 0;
    assign ALU_data_valid_out   = (CTRL_counter_in == 4 && MAC_valid_wr) ? 1 : 0;

    MAC #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) MAC_inst (
        .clk                    (clk            ),
        .rst_n                  (rst_n          ),
        .Filter_in_in           (Filter_in      ),
        .activation_in          (IA_in          ),
        .CTRL_counter_in        (CTRL_counter_in),

        .MAC_out                (MAC_out_wr     ),
        .MAC_valid_out          (MAC_valid_wr   )
    );
endmodule

module MAC #(
    parameter DATA_WIDTH=16,
    parameter ADDR_WIDTH=8
)
(
    input wire                          clk,
    input wire                          rst_n,
    input wire signed [DATA_WIDTH-1:0]  Filter_in,
    input wire signed [DATA_WIDTH-1:0]  IA_in,
    input wire [2:0]                    CTRL_counter_in,

    output reg signed [DATA_WIDTH-1:0]  MAC_out,
    output reg                          MAC_valid_out
);  
    wire signed [2*DATA_WIDTH-1:0] partial_sum0_wr, partial_sum1_wr, partial_sum2_wr, partial_sum3_wr;
    wire signed[2*DATA_WIDTH-1:0] partial_sum4_wr, partial_sum5_wr, partial_sum6_wr, partial_sum7_wr;
    wire signed [2*DATA_WIDTH-1:0] partial_sum8_wr, partial_sum9_wr,partial_sum10_wr, partial_sum11_wr; 
    wire signed[2*DATA_WIDTH-1:0] partial_sum12_wr, partial_sum13_wr, partial_sum14_wr, partial_sum15_wr;
    wire signed [2*DATA_WIDTH-1:0] sum_stage1_wr, sum_stage2_wr, sum_final_wr;

    wire signed [DATA_WIDTH-1:0] multiplicand_wr;
    wire signed [DATA_WIDTH-1:0] multiplier_wr;
    wire                         sign_wr;
    wire signed [DATA_WIDTH-1:0] out_wr;
    wire signed [DATA_WIDTH-1:0] out1_wr;

    reg signed [DATA_WIDTH-1:0] sum_stage1_rg;
    reg signed [DATA_WIDTH-1:0] accumulation_rg;
    reg signed [DATA_WIDTH-1:0] multiplier_rg;
    reg signed [DATA_WIDTH-1:0] multiplicand_rg;
    reg                         sign_rg;
//Clock1
    
    assign multiplicand_wr  =       (Filter_in<0) ? (-Filter_in) : Filter_in;                                         
    assign multiplier_wr    =       (IA_in<0) ? (-IA_in) : IA_in;                                 
    assign sign_wr          =       (Filter_in[DATA_WIDTH-1] ^ IA_in[DATA_WIDTH-1]);                       

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
                                    -$signed(sum_stage2_wr[21:6]+sum_stage2_wr[5:5]); 
    assign out_wr  = sum_final_wr + accumulation_rg;

    assign MAC_out =  out_wr;
    assign MAC_valid_out = (CTRL_counter_in == 4) ? 1 : 0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulation_rg <= 0;
            multiplier_rg   <= 0;
            multiplicand_rg <= 0;
            sum_stage1_rg   <= 0;                                  
            sign_rg         <= 0;
            MAC_out         <= 0;
            MAC_valid_out   <= 0;
        end else begin
            sum_stage1_rg           <= sum_stage1_wr;                                                                                      
            multiplicand_rg         <= multiplicand_wr;
            multiplier_rg           <= multiplier_wr;
            sign_rg                 <= sign_wr;

            if(CTRL_counter_in == 0) begin
                accumulation_rg         <= 0;
            end else begin
                accumulation_rg         <= out_wr; 
            end
        end
    end
endmodule
