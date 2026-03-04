`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 02/08/2026 11:34:50 PM
// Design Name: 
// Module Name: PE
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
module PE #(
    parameter                                  	    DATA_WIDTH=16
)(
	input wire 										clk,
	input wire 										rst_n,
	///*** From the Controller ***///
	input  wire [2:0]                               CTRL_m_count_in,
	input  wire 									load_filter_enable_in			

  	//     //*** From Global Buffer ***///			
	input  wire 					            	IA_valid_in,	    //Input Activation
	input  wire signed [DATA_WIDTH-1:0]             IA_in,			    
	input  wire 					            	Filter_valid_in,	//Weights of filters
	input  wire signed [DATA_WIDTH-1:0]             Filter_in,
	//-----------------------------------------------------//
	//          			Output Signals                 // 
	//-----------------------------------------------------//  
	///*** To Adder Tree ***///
	output wire signed [DATA_WIDTH-1:0]           	PE_out,
	output wire 						           	PE_valid_out
	
);

	// *************** Wire signals *************** //
    reg signed [15:0]         			            fw_mem_rg[4:0];
    reg signed [15:0]         			            ia_mem_rg[4:0];

	wire  					           	  			D0_valid_wr;
	wire signed [DATA_WIDTH-1:0]           	  		D0_wr;


	// *************** Register signals *************** //		
    assign PE_out       = D0_wr;
    assign PE_valid_out = D0_valid_wr;



	ALU alu
	(
		.clk                    (clk                      	 ),
		.rst_n                  (rst_n                    	 ),
		.Filter_in              (fw_mem_rg[CTRL_m_count_in]	 ),
		.IA_in                  (ia_mem_rg[CTRL_m_count_in]	 ),
		.CTRL_m_count_in        (CTRL_m_count_in             ),

		.ALU_data_out           (D0_wr                       ),
		.ALU_data_valid_out     (D0_valid_wr                 )
	);

	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			fw_mem_rg[0] <= 0;
            fw_mem_rg[1] <= 0;
            fw_mem_rg[2] <= 0;
            fw_mem_rg[3] <= 0;
            fw_mem_rg[4] <= 0;

            ia_mem_rg[0] <= 0;
            ia_mem_rg[1] <= 0;
            ia_mem_rg[2] <= 0;
            ia_mem_rg[3] <= 0;
            ia_mem_rg[4] <= 0;
		end	
		else begin			
			if(load_filter_enable_in) begin
                fw_mem_rg[CTRL_m_count_in] <= filter_in;
            end else begin 
                if(CTRL_m_count_in == 4) begin
                    ia_mem_rg[0]               <= ia_mem_rg[1];
                    ia_mem_rg[1]               <= ia_mem_rg[2];
                    ia_mem_rg[2]               <= ia_mem_rg[3];
                    ia_mem_rg[3]               <= ia_mem_rg[4];
                    ia_mem_rg[4]               <= IA_in;
                end 
            end
		end
	end 
endmodule
