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
module PE
#(
    parameter                                  	    DATA_WIDTH=16
)
(
	input  wire                                 	clk,
	input  wire                                 	rst_n,			

	///*** From the Controller ***///	
	input  wire 					              	start_in,

	input  wire [2:0]                               CTRL_s_counter_in,
	input wire 										load_filter_enable_in,	// turn to 1 if change to new group of filer 			
	
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
	reg signed [DATA_WIDTH-1:0]						ia_mem_rg	[0:4];
	reg signed [DATA_WIDTH-1:0]						fw_mem_rg	[0:4];
	wire signed [DATA_WIDTH-1:0]           	  		D0_wr;
	integer i;
	
	// *************** Register signals *************** //		
    assign PE_out       = D0_wr;
								

	
	// Calculation
	ALU alu
	(
		.clk                    (clk                    ),
		.rst_n                  (rst_n          		),
		.ALU_valid_in			(ALU_valid_in			),
		.Filter0_in             (fw_mem_rg[0]			),
		.Filter1_in             (fw_mem_rg[1]			),
		.Filter2_in             (fw_mem_rg[2]			),
		.Filter3_in             (fw_mem_rg[3]			),
		.Filter4_in             (fw_mem_rg[4]			),
		.IA0_in					(ia_mem_rg[0]			),
		.IA1_in					(ia_mem_rg[1]			),
		.IA2_in					(ia_mem_rg[2]			),
		.IA3_in					(ia_mem_rg[3]			),
		.IA4_in					(ia_mem_rg[4]			),

		.ALU_data_out           (D0_wr                  )
	);

	always @(posedge clk or negedge rst_n) begin
		if (!rst_n) begin
			for(i=0;i<5;i=i+1) begin
				fw_mem_rg[i]	<=	0;
				ia_mem_rg[i]	<=	0;
			end
		end	
		else begin			
			if(load_filter_enable_in&&Filter_valid_in) begin
                fw_mem_rg[CTRL_s_counter_in] <= Filter_in;
            end 
            if(IA_valid_in) begin
                ia_mem_rg[0]               <= ia_mem_rg[1];
                ia_mem_rg[1]               <= ia_mem_rg[2];
                ia_mem_rg[2]               <= ia_mem_rg[3];
                ia_mem_rg[3]               <= ia_mem_rg[4];
                ia_mem_rg[4]               <= IA_in;
            end
		end
	end 
endmodule
