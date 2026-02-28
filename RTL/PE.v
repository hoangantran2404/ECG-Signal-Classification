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


`timescale 1ns/1ns

module PE
#(
    parameter                                  	    DATA_WIDTH=16
)
(
	input  wire                                 	clk,
	input  wire                                 	rst_n,			

	///*** From the Controller ***///	
	input  wire 					              	start_in,

	input  wire [2:0]                               CTRL_counter_in,
	input  wire [2:0]       	            		current_state_in,				
	
    //*** From Global Buffer ***///			
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
    reg signed [15:0]         			            fw_mem[4:0];
    reg signed [15:0]         			            IA_mem[4:0];
			    
	wire  					           	  			D0_valid_wr;
	wire signed [DATA_WIDTH-1:0]           	  		D0_wr;

	
	// *************** Register signals *************** //		
    assign PE_out       = D0_wr;
    assign PE_valid_out = D0_valid_wr;
								

	
	// Calculation
	ALU alu
	(
		.clk                    (clk                    ),
		.rst_n                  (rst_n                  ),
		.Filter_in              (fw_mem[CTRL_counter_in]),
		.IA_in                  (IA_mem[CTRL_counter_in]),
		.CTRL_counter_in        (CTRL_counter_in        ),

		.ALU_data_out           (D0_wr                  ),
		.ALU_data_valid_out     (D0_valid_wr            )
	);

	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			fw_mem[0] <= 0;
            fw_mem[1] <= 0;
            fw_mem[2] <= 0;
            fw_mem[3] <= 0;
            fw_mem[4] <= 0;
            
            IA_mem[0] <= 0;
            IA_mem[1] <= 0;
            IA_mem[2] <= 0;
            IA_mem[3] <= 0;
            IA_mem[4] <= 0;
		end	
		else begin			
			if(start_in) begin
                fw_mem[CTRL_counter_in] <= filter_in;
            end else begin 
                if(CTRL_counter_in == 4) begin
                    IA_mem[0]               <= IA_mem[1];
                    IA_mem[1]               <= IA_mem[2];
                    IA_mem[2]               <= IA_mem[3];
                    IA_mem[3]               <= IA_mem[4];
                    IA_mem[4]               <= IA_in;
                end else begin 
                    
                end
            end
		end
	end 
endmodule
