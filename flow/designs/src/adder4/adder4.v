module adder4 (
    input  wire        clk,
    input  wire [3:0]  a,
    input  wire [3:0]  b,
    output reg  [4:0]  sum
);
    always @(posedge clk) begin
        sum <= {1'b0, a} + {1'b0, b};
    end
endmodule
