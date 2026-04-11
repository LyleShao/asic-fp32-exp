// exp_l64_fp32.v
// ---------------------------------------------------------------------------
// FP32 exp(x) implementation based on exp_fp32_l64 (L=64, quartic polynomial)
// Target: ASIC datapath with IEEE-754 single-precision add/mul IP and small ROM
// This is a straightforward, single-module RTL skeleton suitable for synthesis.
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps

module exp_l64_fp32 (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         in_valid,
    input  wire [31:0]  in_x,       // FP32 input x

    output reg          out_valid,
    output reg  [31:0]  out_y       // FP32 output exp(x)
);

    // NOTE: This is a structural skeleton. You must plug in your own
    // FP32 IP cores for add/mul/compare and float<->int conversion.
    // Below we indicate these as function-like modules:
    //   fp32_add, fp32_sub, fp32_mul, fp32_abs, fp32_to_int, int_to_fp32

    // -----------------
    // Stage 0: registers
    // -----------------
    reg        s0_valid;
    reg [31:0] s0_x;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_valid <= 1'b0;
            s0_x     <= 32'h0;
        end else begin
            s0_valid <= in_valid;
            s0_x     <= in_x;
        end
    end

    // For brevity, this skeleton omits explicit NaN/Inf/threshold handling.
    // In a production design, add special-case logic as described in README.

    // -------------------------
    // Stage 1: N, J, M (range reduction)
    // -------------------------
    // N = round(x * INV_L_64), with INV_L_64 = 64 / ln(2) as FP32 constant.

    localparam [31:0] INV_L_64 = 32'h42845F30; // approx 64/log(2) in IEEE FP32
    localparam [31:0] LOG2_BY_64 = 32'h3CB17218; // approx log(2)/64 in IEEE FP32
    localparam [31:0] A2_64 = 32'h3F000000; // ~0.5 (placeholder, tune constants)
    localparam [31:0] A3_64 = 32'h3E2AAAAB; // ~1/6
    localparam [31:0] A4_64 = 32'h3D2AAAAB; // ~1/24

    reg        s1_valid;
    reg [31:0] s1_x;
    reg [31:0] s1_tN;
    reg signed [15:0] s1_N;
    reg [5:0]  s1_J;
    reg signed [15:0] s1_M;

    // FP32 multiply: tN = x * INV_L_64
    wire [31:0] mul1_out;
    fp32_mul u_mul1 (
        .a (s0_x),
        .b (INV_L_64),
        .y (mul1_out)
    );

    // float->int rounding
    wire signed [15:0] to_int_out;
    fp32_to_int u_to_int1 (
        .a  (mul1_out),
        .y  (to_int_out)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_x     <= 32'h0;
            s1_tN    <= 32'h0;
            s1_N     <= 16'sd0;
            s1_J     <= 6'd0;
            s1_M     <= 16'sd0;
        end else begin
            s1_valid <= s0_valid;
            s1_x     <= s0_x;
            s1_tN    <= mul1_out;
            s1_N     <= to_int_out;
            s1_J     <= to_int_out[5:0];              // N mod 64 (simple case)
            s1_M     <= to_int_out >>> 6;             // N >> 6
        end
    end

    // -------------------------
    // Stage 2: reduced argument r = x - N*LOG2_BY_64
    // -------------------------
    reg        s2_valid;
    reg [31:0] s2_r;
    reg [5:0]  s2_J;
    reg signed [15:0] s2_M;

    wire [31:0] s1_N_fp;
    int_to_fp32 u_int_to_fp1 (
        .a (s1_N),
        .y (s1_N_fp)
    );

    wire [31:0] mul2_out;
    fp32_mul u_mul2 (
        .a (s1_N_fp),
        .b (LOG2_BY_64),
        .y (mul2_out)
    );

    wire [31:0] sub1_out;
    fp32_sub u_sub1 (
        .a (s1_x),
        .b (mul2_out),
        .y (sub1_out)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
            s2_r     <= 32'h0;
            s2_J     <= 6'd0;
            s2_M     <= 16'sd0;
        end else begin
            s2_valid <= s1_valid;
            s2_r     <= sub1_out;
            s2_J     <= s1_J;
            s2_M     <= s1_M;
        end
    end

    // -------------------------
    // Stage 3: polynomial p(r)
    // r2 = r*r, r3 = r2*r, r4 = r2*r2, p = r + A2*r2 + A3*r3 + A4*r4
    // -------------------------
    reg        s3_valid;
    reg [31:0] s3_p;
    reg [5:0]  s3_J;
    reg signed [15:0] s3_M;

    wire [31:0] r2, r3, r4;
    wire [31:0] t2, t3, t4;
    wire [31:0] sum23, sum234, sum_p;

    fp32_mul u_mul_r2 (.a(s2_r), .b(s2_r), .y(r2));
    fp32_mul u_mul_r3 (.a(r2),   .b(s2_r), .y(r3));
    fp32_mul u_mul_r4 (.a(r2),   .b(r2),   .y(r4));

    fp32_mul u_mul_t2 (.a(A2_64), .b(r2), .y(t2));
    fp32_mul u_mul_t3 (.a(A3_64), .b(r3), .y(t3));
    fp32_mul u_mul_t4 (.a(A4_64), .b(r4), .y(t4));

    fp32_add u_add_23   (.a(t2),    .b(t3),   .y(sum23));
    fp32_add u_add_234  (.a(sum23), .b(t4),   .y(sum234));
    fp32_add u_add_p    (.a(s2_r),  .b(sum234), .y(sum_p));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_valid <= 1'b0;
            s3_p     <= 32'h0;
            s3_J     <= 6'd0;
            s3_M     <= 16'sd0;
        end else begin
            s3_valid <= s2_valid;
            s3_p     <= sum_p;
            s3_J     <= s2_J;
            s3_M     <= s2_M;
        end
    end

    // -------------------------
    // Stage 4: table lookup and core = T[J] * (1 + p)
    // -------------------------
    reg        s4_valid;
    reg [31:0] s4_core;
    reg signed [15:0] s4_M;

    // 64-entry FP32 ROM for T[J] = 2^(J/64)
    reg [31:0] exp2_table_64 [0:63];
    initial begin
        // NOTE: Fill with the actual FP32 hex values from EXP2_TABLE_64
        // in exp.py. Below are placeholders; replace them before tape-out.
        integer i;
        for (i = 0; i < 64; i = i+1) begin
            exp2_table_64[i] = 32'h3F800000; // 1.0f placeholder
        end
    end

    wire [31:0] T_J = exp2_table_64[s3_J];

    wire [31:0] one_plus_p;
    wire [31:0] core_mul;

    fp32_add u_add_one (
        .a (32'h3F800000), // 1.0f
        .b (s3_p),
        .y (one_plus_p)
    );

    fp32_mul u_mul_core (
        .a (T_J),
        .b (one_plus_p),
        .y (core_mul)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s4_valid <= 1'b0;
            s4_core  <= 32'h0;
            s4_M     <= 16'sd0;
        end else begin
            s4_valid <= s3_valid;
            s4_core  <= core_mul;
            s4_M     <= s3_M;
        end
    end

    // -------------------------
    // Stage 5: scale by 2^M via exponent adjust and pack FP32
    // -------------------------
    reg        s5_valid;
    reg [31:0] s5_y;

    wire [31:0] core_bits = s4_core;
    wire  sign_core = core_bits[31];
    wire [7:0] exp_core  = core_bits[30:23];
    wire [22:0] frac_core = core_bits[22:0];

    // interpret exponent as signed (E_core = exp_core - 127)
    wire signed [9:0] E_core = $signed({1'b0, exp_core}) - 10'sd127;
    wire signed [9:0] E_final = E_core + $signed(s4_M);

    reg [31:0] y_bits;

    always @* begin
        // Simple overflow/underflow handling
        if (E_final > 10'sd127) begin
            // overflow -> +INF
            y_bits = {sign_core, 8'hFF, 23'h0};
        end else if (E_final < -10'sd126) begin
            // underflow -> 0
            y_bits = {sign_core, 8'h00, 23'h0};
        end else begin
            // normal case
            wire [7:0] exp_out = E_final[7:0] + 8'd127;
            y_bits = {sign_core, exp_out, frac_core};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s5_valid <= 1'b0;
            s5_y     <= 32'h0;
        end else begin
            s5_valid <= s4_valid;
            s5_y     <= y_bits;
        end
    end

    // -------------------------
    // Output stage
    // -------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_y     <= 32'h0;
        end else begin
            out_valid <= s5_valid;
            out_y     <= s5_y;
        end
    end

endmodule

// Stub FP32 IPs --------------------------------------------------------
// Replace these with actual implementations or wrappers around your
// technology's FP32 library.

module fp32_mul(input [31:0] a, input [31:0] b, output [31:0] y);
    assign y = 32'h0; // TODO: replace with real FP32 multiplier
endmodule

module fp32_add(input [31:0] a, input [31:0] b, output [31:0] y);
    assign y = 32'h0; // TODO: replace with real FP32 adder
endmodule

module fp32_sub(input [31:0] a, input [31:0] b, output [31:0] y);
    assign y = 32'h0; // TODO: replace with real FP32 subtractor
endmodule

module fp32_to_int(input [31:0] a, output signed [15:0] y);
    assign y = 16'sd0; // TODO: replace with real float->int converter
endmodule

module int_to_fp32(input signed [15:0] a, output [31:0] y);
    assign y = 32'h0; // TODO: replace with real int->float converter
endmodule
