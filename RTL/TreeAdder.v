module TreeAdder #(
    parameter DATA_WIDTH=16
)
(
    input wire                          clk,
    input wire                          rst_n,

    input wire signed [DATA_WIDTH-1:0]  PE0_in,
    input wire signed [DATA_WIDTH-1:0]  PE1_in,
    input wire signed [DATA_WIDTH-1:0]  PE2_in,
    input wire signed [DATA_WIDTH-1:0]  PE3_in,
    input wire signed [DATA_WIDTH-1:0]  PE4_in,
    input wire signed [DATA_WIDTH-1:0]  PE5_in,
    input wire signed [DATA_WIDTH-1:0]  PE6_in,
    input wire signed [DATA_WIDTH-1:0]  PE7_in,
    input wire signed [DATA_WIDTH-1:0]  PE8_in,
    input wire signed [DATA_WIDTH-1:0]  PE9_in,
    input wire signed [DATA_WIDTH-1:0]  PE10_in,
    input wire signed [DATA_WIDTH-1:0]  PE11_in,
    input wire signed [DATA_WIDTH-1:0]  PE12_in,
    input wire signed [DATA_WIDTH-1:0]  PE13_in,
    input wire signed [DATA_WIDTH-1:0]  PE14_in,
    input wire signed [DATA_WIDTH-1:0]  PE15_in,
    input wire                          PE_valid_in,

    output reg signed [DATA_WIDTH-1:0]  TreeAdder_out,
    output reg                          TreeAdder_valid_out
);
    wire signed [DATA_WIDTH-1:0]        sum_final_wr;
    wire signed [DATA_WIDTH-1:0]        sum0_wr;
    wire signed [DATA_WIDTH-1:0]        sum1_wr;
    wire signed [DATA_WIDTH-1:0]        sum2_wr;
    wire signed [DATA_WIDTH-1:0]        sum3_wr;

    reg signed [DATA_WIDTH-1:0]        sum0_rg;
    reg signed [DATA_WIDTH-1:0]        sum1_rg;
    reg signed [DATA_WIDTH-1:0]        sum2_rg;
    reg signed [DATA_WIDTH-1:0]        sum3_rg;

    assign TreeAdder_valid_out = PE_valid_in;

     always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            TreeAdder_out <= 0;
        end else if (TreeAdder_valid_out) begin
            TreeAdder_out <= TreeAdder_wr;
        end
    end

    adder4 adder0 (
        .in0(PE0_in), 
        .in1(PE1_in), 
        .in2(PE2_in), 
        .in3(PE3_in),
        .out(sum0_wr)
    );
    adder4 adder1 (
        .in0(PE4_in), 
        .in1(PE5_in), 
        .in2(PE6_in), 
        .in3(PE7_in),
        .out(sum1_wr)
    );
    adder4 adder2 (
        .in0(PE8_in), 
        .in1(PE9_in), 
        .in2(PE10_in), 
        .in3(PE11_in),
        .out(sum2_wr)
    );
    adder4 adder3 (
        .in0(PE12_in), 
        .in1(PE13_in), 
        .in2(PE14_in), 
        .in3(PE15_in),
        .out(sum3_wr)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum0_rg <= 0; 
            sum1_rg <= 0; 
            sum2_rg <= 0; 
            sum3_rg <= 0;
            valid_stage1_rg <= 0;
        end else begin
            sum0_rg <= sum0_wr;
            sum1_rg <= sum1_wr;
            sum2_rg <= sum2_wr;
            sum3_rg <= sum3_wr;
            valid_stage1_rg <= PE_valid_in; 
        end
    end

    adder4 adder_final (
        .in0(sum0_rg), 
        .in1(sum1_rg), 
        .in2(sum2_rg), 
        .in3(sum3_rg),
        .out(sum_final_wr)
    );
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            TreeAdder_out       <= 0;
            TreeAdder_valid_out <= 0;
        end else begin
            TreeAdder_out       <= sum_final_wr;
            TreeAdder_valid_out <= valid_stage1_rg; 
        end
    end
endmodule 

module adder4 #(
    parameter DATA_WIDTH=16
)
(
    input wire signed [DATA_WIDTH-1:0]  in0,
    input wire signed [DATA_WIDTH-1:0]  in1,
    input wire signed [DATA_WIDTH-1:0]  in2,
    input wire signed [DATA_WIDTH-1:0]  in3,

    output wire signed [DATA_WIDTH-1:0] out
);
    assign out = in0 + in1 + in2 + in3;
endmodule