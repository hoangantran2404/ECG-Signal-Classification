module AccumulatorBuffer #(
    parameter DATA_WIDTH=16
)
(
    input wire                          clk,
    input wire                          rst_n,
    //From Global Buffer
    input wire signed [DATA_WIDTH-1:0]  bias_in,
    input wire                          bias_valid_in,
    // From Tree Adder
    input wire signed [DATA_WIDTH-1:0]  TA0_in,
    input wire signed [DATA_WIDTH-1:0]  TA1_in,
    input wire signed [DATA_WIDTH-1:0]  TA2_in,
    input wire signed [DATA_WIDTH-1:0]  TA3_in,
    input wire signed [DATA_WIDTH-1:0]  TA4_in,
    input wire signed [DATA_WIDTH-1:0]  TA5_in,
    input wire signed [DATA_WIDTH-1:0]  TA6_in,
    input wire signed [DATA_WIDTH-1:0]  TA7_in,
    input wire signed [DATA_WIDTH-1:0]  TA8_in,
    input wire signed [DATA_WIDTH-1:0]  TA9_in,
    input wire signed [DATA_WIDTH-1:0]  TA10_in,
    input wire signed [DATA_WIDTH-1:0]  TA11_in,
    input wire signed [DATA_WIDTH-1:0]  TA12_in,
    input wire signed [DATA_WIDTH-1:0]  TA13_in,
    input wire signed [DATA_WIDTH-1:0]  TA14_in,
    input wire signed [DATA_WIDTH-1:0]  TA15_in,
    input wire                          TA_valid_in,
    // From Controller
    input wire                          ReLU_enable_in,
    input wire [7:0]                    CTRL_s_counter_in,          //Connect to s_counter
    input wire [4:0]                    CTRL_ia_counter_in,         //Connect to i_counter
    input wire [6:0]                    CTRL_fw_counter_in,
    input wire [2:0]                    CTRL_FSM_layer_in,

    //To Global Buffer
    output wire signed [DATA_WIDTH-1:0]  AB0_out,
    output wire signed [DATA_WIDTH-1:0]  AB1_out,
    output wire signed [DATA_WIDTH-1:0]  AB2_out,
    output wire signed [DATA_WIDTH-1:0]  AB3_out,
    output wire signed [DATA_WIDTH-1:0]  AB4_out,
    output wire signed [DATA_WIDTH-1:0]  AB5_out,
    output wire signed [DATA_WIDTH-1:0]  AB6_out,
    output wire signed [DATA_WIDTH-1:0]  AB7_out,
    output wire signed [DATA_WIDTH-1:0]  AB8_out,
    output wire signed [DATA_WIDTH-1:0]  AB9_out,
    output wire signed [DATA_WIDTH-1:0]  AB10_out,
    output wire signed [DATA_WIDTH-1:0]  AB11_out,
    output wire signed [DATA_WIDTH-1:0]  AB12_out,
    output wire signed [DATA_WIDTH-1:0]  AB13_out,
    output wire signed [DATA_WIDTH-1:0]  AB14_out,
    output wire signed [DATA_WIDTH-1:0]  AB15_out,

    output reg                          Accumulator_valid_out
);
    //==================================================//
    //               Internal Signals                   //
    //==================================================//
    wire [7:0] BRAM_read_rg;
    wire signed [DATA_WIDTH-1:0] BRAM_0_out_wr, BRAM_1_out_wr, BRAM_2_out_wr,BRAM_3_out_wr,
                                BRAM_4_out_wr, BRAM_5_out_wr, BRAM_6_out_wr, BRAM_7_out_wr, 
                                BRAM_8_out_wr, BRAM_9_out_wr, BRAM_10_out_wr, BRAM_11_out_wr,
                                BRAM_12_out_wr, BRAM_13_out_wr, BRAM_14_out_wr, BRAM_15_out_wr;
    wire signed [DATA_WIDTH-1:0] BRAM_0_in_wr, BRAM_1_in_wr, BRAM_2_in_wr,BRAM_3_in_wr,
                                BRAM_4_in_wr, BRAM_5_in_wr, BRAM_6_in_wr, BRAM_7_in_wr, 
                                BRAM_8_in_wr, BRAM_9_in_wr, BRAM_10_in_wr, BRAM_11_in_wr,
                                BRAM_12_in_wr, BRAM_13_in_wr, BRAM_14_in_wr, BRAM_15_in_wr;
    wire signed [DATA_WIDTH-1:0] AB0_wr, AB1_wr, AB2_wr, AB3_wr,
                                AB4_wr, AB5_wr, AB6_wr, AB7_wr,
                                AB8_wr, AB9_wr, AB10_wr, AB11_wr,
                                AB12_wr, AB13_wr, AB14_wr, AB15_wr;
    reg signed [DATA_WIDTH-1:0] TA0_rg, TA1_rg,TA2_rg, TA3_rg,
                                TA4_rg, TA5_rg,TA6_rg, TA7_rg,
                                TA8_rg, TA9_rg,TA10_rg, TA11_rg,
                                TA12_rg, TA13_rg,TA14_rg, TA15_rg;
    reg                         TA_valid_rg;
    reg [4:0]                   CTRL_ia_counter_rg;
    reg signed [DATA_WIDTH-1:0] bias_mem [0:15];
    reg [7:0]                   CTRL_counter_B_rg;
    reg [7:0]                   k;
    reg [4:0]                   in_shape;
    integer i;
    
    //==================================================//
    //               Combinational Logic                //
    //==================================================//
    assign BRAM_0_in_wr = (CTRL_ia_counter_rg == 0)? TA0_rg :  BRAM_0_out_wr + TA0_rg; 
    assign BRAM_1_in_wr = (CTRL_ia_counter_rg == 0)? TA1_rg :  BRAM_1_out_wr + TA1_rg;
    assign BRAM_2_in_wr = (CTRL_ia_counter_rg == 0)? TA2_rg :  BRAM_2_out_wr + TA2_rg;
    assign BRAM_3_in_wr = (CTRL_ia_counter_rg == 0)? TA3_rg :  BRAM_3_out_wr + TA3_rg;
    assign BRAM_4_in_wr = (CTRL_ia_counter_rg == 0)? TA4_rg :  BRAM_4_out_wr + TA4_rg;
    assign BRAM_5_in_wr = (CTRL_ia_counter_rg == 0)? TA5_rg :  BRAM_5_out_wr + TA5_rg;
    assign BRAM_6_in_wr = (CTRL_ia_counter_rg == 0)? TA6_rg :  BRAM_6_out_wr + TA6_rg;
    assign BRAM_7_in_wr = (CTRL_ia_counter_rg == 0)? TA7_rg :  BRAM_7_out_wr + TA7_rg;
    assign BRAM_8_in_wr = (CTRL_ia_counter_rg == 0)? TA8_rg :  BRAM_8_out_wr + TA8_rg;
    assign BRAM_9_in_wr = (CTRL_ia_counter_rg == 0)? TA9_rg :  BRAM_9_out_wr + TA9_rg;
    assign BRAM_10_in_wr = (CTRL_ia_counter_rg == 0)? TA10_rg :  BRAM_10_out_wr + TA10_rg;
    assign BRAM_11_in_wr = (CTRL_ia_counter_rg == 0)? TA11_rg :  BRAM_11_out_wr + TA11_rg;
    assign BRAM_12_in_wr = (CTRL_ia_counter_rg == 0)? TA12_rg :  BRAM_12_out_wr + TA12_rg;
    assign BRAM_13_in_wr = (CTRL_ia_counter_rg == 0)? TA13_rg :  BRAM_13_out_wr + TA13_rg;
    assign BRAM_14_in_wr = (CTRL_ia_counter_rg == 0)? TA14_rg :  BRAM_14_out_wr + TA14_rg;
    assign BRAM_15_in_wr = (CTRL_ia_counter_rg == 0)? TA15_rg :  BRAM_15_out_wr + TA15_rg;
// Plus bias
    assign AB0_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_0_in_wr) + bias_mem[0]:16'd0;
    assign AB1_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_1_in_wr) + bias_mem[1]:16'd0;
    assign AB2_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_2_in_wr) + bias_mem[2]:16'd0;
    assign AB3_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_3_in_wr) + bias_mem[3]:16'd0;
    assign AB4_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_4_in_wr) + bias_mem[4]:16'd0;
    assign AB5_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_5_in_wr) + bias_mem[5]:16'd0;
    assign AB6_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_6_in_wr) + bias_mem[6]:16'd0;
    assign AB7_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_7_in_wr) + bias_mem[7]:16'd0;
    assign AB8_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_8_in_wr) + bias_mem[8]:16'd0;
    assign AB9_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_9_in_wr) + bias_mem[9]:16'd0;
    assign AB10_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_10_in_wr) + bias_mem[10]:16'd0;
    assign AB11_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_11_in_wr) + bias_mem[11]:16'd0;
    assign AB12_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_12_in_wr) + bias_mem[12]:16'd0;
    assign AB13_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_13_in_wr) + bias_mem[13]:16'd0;
    assign AB14_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_14_in_wr) + bias_mem[14]:16'd0;
    assign AB15_wr  = (CTRL_ia_counter_rg==in_shape)? $signed(BRAM_15_in_wr) + bias_mem[15]:16'd0;

// Write output
    assign AB0_out  = (ReLU_enable_in && $signed(AB0_wr)<0 )? 16'd0: AB0_wr;
    assign AB1_out  = (ReLU_enable_in && $signed(AB1_wr)<0 )? 16'd0: AB1_wr;
    assign AB2_out  = (ReLU_enable_in && $signed(AB2_wr)<0 )? 16'd0: AB2_wr;
    assign AB3_out  = (ReLU_enable_in && $signed(AB3_wr)<0 )? 16'd0: AB3_wr;
    assign AB4_out  = (ReLU_enable_in && $signed(AB4_wr)<0 )? 16'd0: AB4_wr;
    assign AB5_out  = (ReLU_enable_in && $signed(AB5_wr)<0 )? 16'd0: AB5_wr;
    assign AB6_out  = (ReLU_enable_in && $signed(AB6_wr)<0 )? 16'd0: AB6_wr;
    assign AB7_out  = (ReLU_enable_in && $signed(AB7_wr)<0 )? 16'd0: AB7_wr;
    assign AB8_out  = (ReLU_enable_in && $signed(AB8_wr)<0 )? 16'd0: AB8_wr;
    assign AB9_out  = (ReLU_enable_in && $signed(AB9_wr)<0 )? 16'd0: AB9_wr;
    assign AB10_out  = (ReLU_enable_in && $signed(AB10_wr)<0 )? 16'd0: AB10_wr;
    assign AB11_out  = (ReLU_enable_in && $signed(AB11_wr)<0 )? 16'd0: AB11_wr;
    assign AB12_out  = (ReLU_enable_in && $signed(AB12_wr)<0 )? 16'd0: AB12_wr;
    assign AB13_out  = (ReLU_enable_in && $signed(AB13_wr)<0 )? 16'd0: AB13_wr;
    assign AB14_out  = (ReLU_enable_in && $signed(AB14_wr)<0 )? 16'd0: AB14_wr;
    assign AB15_out  = (ReLU_enable_in && $signed(AB15_wr)<0 )? 16'd0: AB15_wr;

    //==================================================//
    //                Instantiate module                //
    //==================================================//
// Port A for reading only, Port B for writing only
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter0 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_0_out_wr      ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_0_in_wr       ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter1 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_1_out_wr      ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg    ),
        .dinb           (BRAM_1_in_wr       ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter2 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_2_out_wr      ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_2_in_wr       ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter3 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_3_out_wr       ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_3_in_wr       ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter4 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_4_out_wr      ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_4_in_wr       ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter5 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_5_out_wr      ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_5_out_wr      ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter6 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_6_out_wr      ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_6_in_wr       ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter7 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_7_out_wr      ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_7_in_wr       ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter8 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_8_out_wr      ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_8_in_wr       ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter9 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_9_out_wr      ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_9_in_wr       ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter10 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_10_out_wr     ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_10_in_wr      ),
        .doutb          (                   )   
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter11 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_11_out_wr     ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_11_in_wr      ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter12 (
        .clk            (clk                ),
        .wea            (0                  ), 
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ), 
        .dina           (0                  ),
        .douta          (BRAM_12_out_wr     ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_12_in_wr      ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter13 (
        .clk            (clk                ),
        .wea            (0                  ),
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ),
        .dina           (0                  ),
        .douta          (BRAM_13_out_wr     ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_13_in_wr      ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter14 (
        .clk            (clk                ),
        .wea            (0                  ),
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ),
        .dina           (0                  ),
        .douta          (BRAM_14_out_wr     ),

        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_14_in_wr      ),
        .doutb          (                   )
    );
    BRAM1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(8)
    ) filter15 (
        .clk            (clk                ),
        .wea            (0                  ),
        .ena            (1                  ),
        .addra          (CTRL_s_counter_in  ),
        .dina           (0                  ),
        .douta          (BRAM_15_out_wr     ),
        
        .web            (TA_valid_in        ),
        .enb            (TA_valid_in        ),
        .addrb          (CTRL_counter_B_rg  ),
        .dinb           (BRAM_15_in_wr      ),
        .doutb          (                   )
    );
    always @(posedge clk or negedge rst_n) begin
        case(CTRL_FSM_layer_in)
            2'd0: begin
                k           = 191;                            //K is the number of sliding windoes
                in_shape    = 0;
            end                                     
            2'd1: begin
                k           = 199;
                in_shape    = 3;                            // 64 input_channels
            end
            2'd2: begin
                k           = 215;
                in_shape    = 7;                            //128 input_channels
            end
            2'd3: begin 
                k           = 247;
                in_shape    = 15;                           //256 input_channels
            end
            default: begin
                k = 0;
                in_shape = 0;
            end
        endcase
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            CTRL_counter_B_rg   <= 0;
            CTRL_ia_counter_rg  <= 0;
            TA_valid_rg         <= 0;

            TA0_rg              <= 0;
            TA1_rg              <= 0;
            TA2_rg              <= 0;
            TA3_rg              <= 0;
            TA4_rg              <= 0;
            TA5_rg              <= 0;
            TA6_rg              <= 0;
            TA7_rg              <= 0;
            TA8_rg              <= 0;
            TA9_rg              <= 0;
            TA10_rg             <= 0;
            TA11_rg             <= 0;
            TA12_rg             <= 0;
            TA13_rg             <= 0;
            TA14_rg             <= 0;
            TA15_rg             <= 0;

            for(i=0;i<16;i=i+1) begin
                bias_mem[i]     <= 16'd0;
            end 
        end else begin
            CTRL_counter_B_rg   <= CTRL_s_counter_in;

            TA0_rg              <= TA0_in;
            TA1_rg              <= TA1_in;
            TA2_rg              <= TA2_in;
            TA3_rg              <= TA3_in;
            TA4_rg              <= TA4_in;
            TA5_rg              <= TA5_in;
            TA6_rg              <= TA6_in;
            TA7_rg              <= TA7_in;
            TA8_rg              <= TA8_in;
            TA9_rg              <= TA9_in;
            TA10_rg             <= TA10_in;
            TA11_rg             <= TA11_in;
            TA12_rg             <= TA12_in;
            TA13_rg             <= TA13_in;
            TA14_rg             <= TA14_in;
            TA15_rg             <= TA15_in;

            CTRL_ia_counter_rg <= CTRL_ia_counter_in;
            TA_valid_rg        <= TA_valid_in;

            if(CTRL_ia_counter_in ==0 && CTRL_counter_B_rg< 8'd16) begin 
                    bias_mem[CTRL_counter_B_rg] <= bias_in;
            end 

            if (CTRL_ia_counter_in == in_shape && TA_valid_in) begin
                Accumulator_valid_out <= 1'b1;
            end else begin
                Accumulator_valid_out <= 1'b0;
            end
        end
    end
endmodule