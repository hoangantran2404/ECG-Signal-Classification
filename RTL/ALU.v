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
    input wire                          ALU_valid_in,
    input wire signed [DATA_WIDTH-1:0]  Filter0_in,
    input wire signed [DATA_WIDTH-1:0]  Filter1_in,
    input wire signed [DATA_WIDTH-1:0]  Filter2_in,
    input wire signed [DATA_WIDTH-1:0]  Filter3_in,
    input wire signed [DATA_WIDTH-1:0]  Filter4_in,
    input wire signed [DATA_WIDTH-1:0]  IA0_in,
    input wire signed [DATA_WIDTH-1:0]  IA1_in,
    input wire signed [DATA_WIDTH-1:0]  IA2_in,
    input wire signed [DATA_WIDTH-1:0]  IA3_in,
    input wire signed [DATA_WIDTH-1:0]  IA4_in,             

    output wire signed [DATA_WIDTH-1:0] ALU_data_out,
    output reg                         ALU_data_valid_out
);
    wire signed [DATA_WIDTH-1:0]        MAC0_out_wr, MAC1_out_wr, MAC2_out_wr, MAC3_out_wr, MAC4_out_wr;
    wire signed [DATA_WIDTH-1:0]        adder_wr;
    
    assign ALU_data_out         =       adder_wr + MAC4_out_wr;

    MAC #(
        .DATA_WIDTH(DATA_WIDTH)
    ) MAC_0 (
        .clk                    (clk            ),
        .rst_n                  (rst_n          ),
        .Filter_in              (Filter0_in     ),
        .IA_in                  (IA0_in         ),

        .MAC_out                (MAC0_out_wr    )
    );
    MAC #(
        .DATA_WIDTH(DATA_WIDTH)
    ) MAC_1 (
        .clk                    (clk            ),
        .rst_n                  (rst_n          ),
        .Filter_in              (Filter1_in     ),
        .IA_in                  (IA1_in         ),

        .MAC_out                (MAC1_out_wr     )
    );
    MAC #(
        .DATA_WIDTH(DATA_WIDTH)
    ) MAC_2 (
        .clk                    (clk            ),
        .rst_n                  (rst_n          ),
        .Filter_in              (Filter2_in     ),
        .IA_in                  (IA2_in         ),

        .MAC_out                (MAC2_out_wr     )
    );
    MAC #(
        .DATA_WIDTH(DATA_WIDTH)
    ) MAC_3 (
        .clk                    (clk            ),
        .rst_n                  (rst_n          ),
        .Filter_in              (Filter3_in     ),
        .IA_in                  (IA3_in         ),

        .MAC_out                (MAC3_out_wr     )
    );
    MAC #(
        .DATA_WIDTH(DATA_WIDTH)
    ) MAC_4 (
        .clk                    (clk            ),
        .rst_n                  (rst_n          ),
        .Filter_in              (Filter4_in     ),
        .IA_in                  (IA4_in         ),

        .MAC_out                (MAC4_out_wr     )
    );

    adder4 adder(
        .in0(MAC0_out_wr),
        .in1(MAC1_out_wr),
        .in2(MAC2_out_wr),
        .in3(MAC3_out_wr),
        .out(adder_wr)
    );
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) 
            ALU_data_valid_out <= 1'b0;
        else 
            ALU_data_valid_out <= ALU_valid_in; 
    end
endmodule

module MAC #(
    parameter DATA_WIDTH=16
)
(
    input wire                          clk,
    input wire                          rst_n,
    input wire signed [DATA_WIDTH-1:0]  Filter_in,
    input wire signed [DATA_WIDTH-1:0]  IA_in,

    output reg signed [DATA_WIDTH-1:0]  MAC_out
);  

    wire signed [2*DATA_WIDTH-1:0]  product_wr;
    wire signed [DATA_WIDTH-1:0]    product_final_wr;

    reg signed [DATA_WIDTH-1:0]     multiplier_rg;
    reg signed [DATA_WIDTH-1:0]     multiplicand_rg;              

    assign product_wr       =       $signed(multiplicand_rg) * $signed(multiplier_rg);

    assign product_final_wr =       $signed(product_wr[21:6]) + $signed(product_wr[5:5]);


    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            multiplier_rg           <= 0;
            multiplicand_rg         <= 0;  
            MAC_out                 <= 0;                   
        end else begin                                                                                    
            multiplicand_rg         <= IA_in;
            multiplier_rg           <= Filter_in;
            MAC_out                 <= product_final_wr;
        end
    end
endmodule
