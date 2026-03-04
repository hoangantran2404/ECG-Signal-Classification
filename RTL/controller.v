`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 01/25/2026 03:17:11 PM
// Design Name: 
// Module Name: controller
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
module Controller #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 6
) (
    input wire              clk,
    input wire              rst_n,
    // From Global Controller
    input wire              start_in,
    input wire              kernel_size_in,
    input wire              last_layer_in,
    input wire [7:0]        slide_num_in,
    input wire [4:0]        in_shape_in,
    input wire [4:0]        out_shape_in,
    // From Global Buffer

    //To Global Controller
    output wire             layer_done_out,
    //TO PE 
    output wire             load_filter_enable_out
    output [3:0]            CTRL_m_count_out,
    //To Accumulator Buffer
    output wire             load_bias_enable_out,
    output wire             send_ofmap_enable_out,
    output wire             ReLU_enable_out,
    output wire [7:0]       CTRL_s_counter_out
   
);  
reg  [2:0]  current_state_rg;
reg  [2:0]  next_state_rg;   
wire        ReLU_enable_wr;

reg [7:0]   slide_num_rg;
reg [4:0]   in_shape_rg;
reg [4:0]   out_shape_rg;

reg [3:0]   m_count_rg;                 // count the multiplication in 1 kernel_size
reg [7:0]   s_count_rg;                 // count the number of sliding window
reg [4:0]   ia_count_rg;                // count the number of Activation Group
reg [4:0]   fw_count_rg;                // count the number of Filter Group

assign load_filter_enable_out   = (current_state_rg==s_LOAD);
assign layer_done_out           = layer_done_rg;
assign load_bias_enable_out     = (current_state_rg == s_LOAD && ia_count_rg == 0 && m_count_rg == 0);
assign send_ofmap_enable_out    = (current_state_rg==s_EXEC&&ia_count_rg==in_shape_in-1);
assign ReLU_enable_wr           = send_ofmap_enable_out&&!last_layer_in;
// assign save_ofmap_done_wr       = (s_count_rg==slide_num_in);
//Them Logic cho filter_valid_in va IA_valid_in

localparam s_IDLE           =   3'b000;
localparam s_LOAD           =   3'b001;
localparam s_EXEC           =   3'b010;
localparam s_DONE           =   3'b011;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        current_state_rg    <=  s_IDLE;
    end else begin
        current_state_rg    <=  next_state_rg;
    end
end
always @(posedge clk or negedge rst_n) begin
    case(current_state_rg):
        s_IDLE: begin 
            if(start_in) begin 
                next_state_rg   =  s_EXEC;
            end else begin
                next_state_rg   =  s_IDLE;
            end
        end
        s_LOAD: begin
            if(m_count_rg == kernel_size_in-1) begin
                next_state_rg   = s_EXEC;
            end else begin
                next_state_rg   =  s_LOAD;
            end
        end
        s_EXEC: begin 
            if (m_count_rg == kernel_size_in - 1) begin
                    if (s_count_rg == slide_num_in - 1) begin
                        if (ia_count_rg == in_shape_in - 1) begin
                            if (fw_count_rg == out_shape_in - 1) begin
                                next_state_rg = s_DONE; 
                            end else begin
                                next_state_rg = s_LOAD; 
                            end
                        end else begin
                            next_state_rg = s_LOAD; 
                        end
                    end
            end
        end
        s_DONE: 
            next_state_rg   <=  s_IDLE;
    endcase
end
//-----------------------------------------------------//
//                       Datapath                      // 
//-----------------------------------------------------// 
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        m_count_rg      <=  0;
        s_count_rg      <=  0;
        ia_count_rg     <=  0;
        fw_count_rg     <=  0;
        layer_done_rg   <=  0;
    end else begin
        case (current_state_rg):
            s_IDLE: begin
                m_count_rg      <=  0;
                s_count_rg      <=  0;
                ia_count_rg     <=  0;
                fw_count_rg     <=  0;
                layer_done_rg   <=  0;
            end
            s_EXEC: begin
                if(m_count_rg == kernel_size_in-1) begin
                    m_count_rg  <= 0;
                    if(s_count_rg   ==  slide_num_in-1) begin
                        s_count_rg <= 0;
                        if(ia_count_rg   ==  in_shape_in-1) begin
                            ia_count_rg      <=   0;
                            if(fw_count_rg == out_shape_in-1) begin
                                fw_count_rg     <=  0;
                                layer_done_rg   <=  1;
                            end else begin
                                fw_count_rg <=   fw_count_rg + 1;
                            end
                        end else begin
                            ia_count_rg  <= ia_count_rg   +   1;
                        end
                    end else begin
                        s_count_rg  <=  s_count_rg + 1;
                    end
                end else begin 
                    m_count_rg  <=  m_count_rg + 1;
                end
            end
            s_DONE:
                layer_done_out <= 0;

        endcase
    end
end
endmodule
