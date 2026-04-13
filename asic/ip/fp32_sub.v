// fp32_sub.v
// ---------------------------------------------------------------------------
// IEEE-754 single-precision FP32 subtractor.
// Combinational, Verilog-2001, Yosys-synthesizable.
//
// Ports:
//   a [31:0]  – FP32 operand A
//   b [31:0]  – FP32 operand B
//   y [31:0]  – FP32 result = a - b
//
// Implementation: flip the sign of b and delegate to fp32_add.
// All special cases (Inf-Inf→NaN, NaN propagation, x-0, etc.)
// are handled correctly by fp32_add.
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps

module fp32_sub (
    input  [31:0] a,
    input  [31:0] b,
    output [31:0] y
);
    wire [31:0] b_neg = {~b[31], b[30:0]};

    fp32_add u_add (
        .a(a),
        .b(b_neg),
        .y(y)
    );

endmodule
