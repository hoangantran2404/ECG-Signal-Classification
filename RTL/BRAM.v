module BRAM1#(
    parameter DATA_WIDTH=32,
    parameter ADDR_WIDTH=10
)(
    input wire clk,
    input wire wea,ena,
    input wire [ADDR_WIDTH-1:0] addra,
    input wire [DATA_WIDTH-1:0] dina,

    input wire web,enb,
    input wire [ADDR_WIDTH-1:0] addrb,
    input wire [DATA_WIDTH-1:0] dinb,

    output reg [DATA_WIDTH-1:0] douta,
    output reg [DATA_WIDTH-1:0] doutb
);
    reg [DATA_WIDTH-1:0] mem [2**ADDR_WIDTH-1:0];

    always @(posedge clk) begin
        if (ena&enb) begin
            if(wea) begin
                mem[addra] <= dina;
            end
            if(web) begin
                mem[addrb] <= dinb;
            end
                douta <= mem[addra];
                doutb <= mem[addrb];
        end else if(ena) begin
            if(wea) begin
                mem[addra] <= dina;
            end
            douta <= mem[addra];
        end else if(enb) begin
            if(web) begin
                mem[addrb] <= dinb;
            end
            doutb <= mem[addrb];
        end else begin
            douta <= 0;
            doutb <= 0;
        end
    end
endmodule

module BRAM2#(
    parameter DATA_WIDTH=32,
    parameter ADDR_WIDTH=10
)(
    input wire clk,
    input wire wea,ena,
    input wire [ADDR_WIDTH-1:0] addra,
    input wire [DATA_WIDTH-1:0] dina,

    input wire web,enb,
    input wire [ADDR_WIDTH-1:0] addrb,
    input wire [DATA_WIDTH-1:0] dinb,

    output reg [DATA_WIDTH-1:0] douta,
    output reg [DATA_WIDTH-1:0] doutb
);
    reg [DATA_WIDTH-1:0] mem [2**ADDR_WIDTH-1:0];

    always @(posedge clk) begin
        if (ena&enb) begin
            if(wea) begin
                mem[addra] <= dina;
            end
            if(web) begin
                mem[addrb] <= dinb;
            end
                douta <= mem[addra];
                doutb <= mem[addrb];
        end else if(ena) begin
            if(wea) begin
                mem[addra] <= dina;
            end
            douta <= mem[addra];
        end else if(enb) begin
            if(web) begin
                mem[addrb] <= dinb;
            end
            doutb <= mem[addrb];
        end else begin
            douta <= 0;
            doutb <= 0;
        end
    end
endmodule
