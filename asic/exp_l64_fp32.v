// exp_l64_fp32.v
// ---------------------------------------------------------------------------
// FP32 exp(x) implementation based on exp_fp32_l64 (L=64, quartic polynomial)
// Target: ASIC datapath with IEEE-754 single-precision add/mul IP and small ROM
//
// TIMING CONSTRAINT: Only 1 FP32 IP operation per cycle for timing closure.
// This results in a deep pipeline (~18 stages) with buffering after each FP op.
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

    // Constants from Python exp.py
    localparam [31:0] INV_L_64    = 32'h42B8AA3B; // 64/log(2) ≈ 92.332
    localparam [31:0] LOG2_BY_64  = 32'h3C317218; // log(2)/64 ≈ 0.01083
    localparam [31:0] A2_64       = 32'h3F000000; // 0.5
    localparam [31:0] A3_64       = 32'h3E2AAAB7; // ~1/6
    localparam [31:0] A4_64       = 32'h3D2AAABD; // ~1/24
    localparam [31:0] FP32_ONE    = 32'h3F800000; // 1.0

    // 64-entry ROM for T[J] = 2^(J/64)
    reg [31:0] exp2_table_64 [0:63];
    initial begin
        exp2_table_64[ 0] = 32'h3F800000; exp2_table_64[ 1] = 32'h3F8164D2;
        exp2_table_64[ 2] = 32'h3F82CD87; exp2_table_64[ 3] = 32'h3F843A29;
        exp2_table_64[ 4] = 32'h3F85AAC3; exp2_table_64[ 5] = 32'h3F871F62;
        exp2_table_64[ 6] = 32'h3F88980F; exp2_table_64[ 7] = 32'h3F8A14D5;
        exp2_table_64[ 8] = 32'h3F8B95C2; exp2_table_64[ 9] = 32'h3F8D1ADF;
        exp2_table_64[10] = 32'h3F8EA43A; exp2_table_64[11] = 32'h3F9031DC;
        exp2_table_64[12] = 32'h3F91C3D3; exp2_table_64[13] = 32'h3F935A2B;
        exp2_table_64[14] = 32'h3F94F4F0; exp2_table_64[15] = 32'h3F96942D;
        exp2_table_64[16] = 32'h3F9837F0; exp2_table_64[17] = 32'h3F99E046;
        exp2_table_64[18] = 32'h3F9B8D3A; exp2_table_64[19] = 32'h3F9D3EDA;
        exp2_table_64[20] = 32'h3F9EF532; exp2_table_64[21] = 32'h3FA0B051;
        exp2_table_64[22] = 32'h3FA27043; exp2_table_64[23] = 32'h3FA43516;
        exp2_table_64[24] = 32'h3FA5FED7; exp2_table_64[25] = 32'h3FA7CD94;
        exp2_table_64[26] = 32'h3FA9A15B; exp2_table_64[27] = 32'h3FAB7A3A;
        exp2_table_64[28] = 32'h3FAD583F; exp2_table_64[29] = 32'h3FAF3B79;
        exp2_table_64[30] = 32'h3FB123F6; exp2_table_64[31] = 32'h3FB311C4;
        exp2_table_64[32] = 32'h3FB504F3; exp2_table_64[33] = 32'h3FB6FD92;
        exp2_table_64[34] = 32'h3FB8FBAF; exp2_table_64[35] = 32'h3FBAFF5B;
        exp2_table_64[36] = 32'h3FBD08A4; exp2_table_64[37] = 32'h3FBF179A;
        exp2_table_64[38] = 32'h3FC12C4D; exp2_table_64[39] = 32'h3FC346CD;
        exp2_table_64[40] = 32'h3FC5672A; exp2_table_64[41] = 32'h3FC78D75;
        exp2_table_64[42] = 32'h3FC9B9BE; exp2_table_64[43] = 32'h3FCBEC15;
        exp2_table_64[44] = 32'h3FCE248C; exp2_table_64[45] = 32'h3FD06334;
        exp2_table_64[46] = 32'h3FD2A81E; exp2_table_64[47] = 32'h3FD4F35B;
        exp2_table_64[48] = 32'h3FD744FD; exp2_table_64[49] = 32'h3FD99D16;
        exp2_table_64[50] = 32'h3FDBFBB8; exp2_table_64[51] = 32'h3FDE60F5;
        exp2_table_64[52] = 32'h3FE0CCDF; exp2_table_64[53] = 32'h3FE33F89;
        exp2_table_64[54] = 32'h3FE5B907; exp2_table_64[55] = 32'h3FE8396A;
        exp2_table_64[56] = 32'h3FEAC0C7; exp2_table_64[57] = 32'h3FED4F30;
        exp2_table_64[58] = 32'h3FEFE4BA; exp2_table_64[59] = 32'h3FF28177;
        exp2_table_64[60] = 32'h3FF5257D; exp2_table_64[61] = 32'h3FF7D0DF;
        exp2_table_64[62] = 32'h3FFA83B3; exp2_table_64[63] = 32'h3FFD3E0C;
    end

    // -------------------------------------------------------------------------
    // Stage 0: Input register
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Stage 1: tN = x * INV_L_64
    // -------------------------------------------------------------------------
    reg        s1_valid;
    reg [31:0] s1_x;
    reg [31:0] s1_tN;

    wire [31:0] mul1_out;
    fp32_mul u_mul1 (
        .a (s0_x),
        .b (INV_L_64),
        .y (mul1_out)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_x     <= 32'h0;
            s1_tN    <= 32'h0;
        end else begin
            s1_valid <= s0_valid;
            s1_x     <= s0_x;
            s1_tN    <= mul1_out;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 2: N = round(tN), J = N mod 64, M = N >> 6
    // -------------------------------------------------------------------------
    reg        s2_valid;
    reg [31:0] s2_x;
    reg signed [15:0] s2_N;
    reg [5:0]  s2_J;
    reg signed [15:0] s2_M;

    wire signed [15:0] to_int_out;
    fp32_to_int u_to_int1 (
        .a (s1_tN),
        .y (to_int_out)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
            s2_x     <= 32'h0;
            s2_N     <= 16'sd0;
            s2_J     <= 6'd0;
            s2_M     <= 16'sd0;
        end else begin
            s2_valid <= s1_valid;
            s2_x     <= s1_x;
            s2_N     <= to_int_out;
            s2_J     <= to_int_out[5:0];
            s2_M     <= to_int_out >>> 6;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 3: N_fp = int_to_fp32(N)
    // -------------------------------------------------------------------------
    reg        s3_valid;
    reg [31:0] s3_x;
    reg [31:0] s3_N_fp;
    reg [5:0]  s3_J;
    reg signed [15:0] s3_M;

    wire [31:0] int_to_fp_out;
    int_to_fp32 u_int_to_fp1 (
        .a (s2_N),
        .y (int_to_fp_out)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_valid <= 1'b0;
            s3_x     <= 32'h0;
            s3_N_fp  <= 32'h0;
            s3_J     <= 6'd0;
            s3_M     <= 16'sd0;
        end else begin
            s3_valid <= s2_valid;
            s3_x     <= s2_x;
            s3_N_fp  <= int_to_fp_out;
            s3_J     <= s2_J;
            s3_M     <= s2_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 4: t = N_fp * LOG2_BY_64
    // -------------------------------------------------------------------------
    reg        s4_valid;
    reg [31:0] s4_x;
    reg [31:0] s4_t;
    reg [5:0]  s4_J;
    reg signed [15:0] s4_M;

    wire [31:0] mul2_out;
    fp32_mul u_mul2 (
        .a (s3_N_fp),
        .b (LOG2_BY_64),
        .y (mul2_out)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s4_valid <= 1'b0;
            s4_x     <= 32'h0;
            s4_t     <= 32'h0;
            s4_J     <= 6'd0;
            s4_M     <= 16'sd0;
        end else begin
            s4_valid <= s3_valid;
            s4_x     <= s3_x;
            s4_t     <= mul2_out;
            s4_J     <= s3_J;
            s4_M     <= s3_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 5: r = x - t
    // -------------------------------------------------------------------------
    reg        s5_valid;
    reg [31:0] s5_r;
    reg [5:0]  s5_J;
    reg signed [15:0] s5_M;

    wire [31:0] sub1_out;
    fp32_sub u_sub1 (
        .a (s4_x),
        .b (s4_t),
        .y (sub1_out)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s5_valid <= 1'b0;
            s5_r     <= 32'h0;
            s5_J     <= 6'd0;
            s5_M     <= 16'sd0;
        end else begin
            s5_valid <= s4_valid;
            s5_r     <= sub1_out;
            s5_J     <= s4_J;
            s5_M     <= s4_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 6: r2 = r * r
    // -------------------------------------------------------------------------
    reg        s6_valid;
    reg [31:0] s6_r;
    reg [31:0] s6_r2;
    reg [5:0]  s6_J;
    reg signed [15:0] s6_M;

    wire [31:0] mul_r2;
    fp32_mul u_mul_r2 (
        .a (s5_r),
        .b (s5_r),
        .y (mul_r2)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s6_valid <= 1'b0;
            s6_r     <= 32'h0;
            s6_r2    <= 32'h0;
            s6_J     <= 6'd0;
            s6_M     <= 16'sd0;
        end else begin
            s6_valid <= s5_valid;
            s6_r     <= s5_r;
            s6_r2    <= mul_r2;
            s6_J     <= s5_J;
            s6_M     <= s5_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 7: r3 = r2 * r
    // -------------------------------------------------------------------------
    reg        s7_valid;
    reg [31:0] s7_r;
    reg [31:0] s7_r2;
    reg [31:0] s7_r3;
    reg [5:0]  s7_J;
    reg signed [15:0] s7_M;

    wire [31:0] mul_r3;
    fp32_mul u_mul_r3 (
        .a (s6_r2),
        .b (s6_r),
        .y (mul_r3)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s7_valid <= 1'b0;
            s7_r     <= 32'h0;
            s7_r2    <= 32'h0;
            s7_r3    <= 32'h0;
            s7_J     <= 6'd0;
            s7_M     <= 16'sd0;
        end else begin
            s7_valid <= s6_valid;
            s7_r     <= s6_r;
            s7_r2    <= s6_r2;
            s7_r3    <= mul_r3;
            s7_J     <= s6_J;
            s7_M     <= s6_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 8: r4 = r2 * r2
    // -------------------------------------------------------------------------
    reg        s8_valid;
    reg [31:0] s8_r;
    reg [31:0] s8_r2;
    reg [31:0] s8_r3;
    reg [31:0] s8_r4;
    reg [5:0]  s8_J;
    reg signed [15:0] s8_M;

    wire [31:0] mul_r4;
    fp32_mul u_mul_r4 (
        .a (s7_r2),
        .b (s7_r2),
        .y (mul_r4)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s8_valid <= 1'b0;
            s8_r     <= 32'h0;
            s8_r2    <= 32'h0;
            s8_r3    <= 32'h0;
            s8_r4    <= 32'h0;
            s8_J     <= 6'd0;
            s8_M     <= 16'sd0;
        end else begin
            s8_valid <= s7_valid;
            s8_r     <= s7_r;
            s8_r2    <= s7_r2;
            s8_r3    <= s7_r3;
            s8_r4    <= mul_r4;
            s8_J     <= s7_J;
            s8_M     <= s7_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 9: t2 = A2_64 * r2
    // -------------------------------------------------------------------------
    reg        s9_valid;
    reg [31:0] s9_r;
    reg [31:0] s9_r3;
    reg [31:0] s9_r4;
    reg [31:0] s9_t2;
    reg [5:0]  s9_J;
    reg signed [15:0] s9_M;

    wire [31:0] mul_t2;
    fp32_mul u_mul_t2 (
        .a (A2_64),
        .b (s8_r2),
        .y (mul_t2)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s9_valid <= 1'b0;
            s9_r     <= 32'h0;
            s9_r3    <= 32'h0;
            s9_r4    <= 32'h0;
            s9_t2    <= 32'h0;
            s9_J     <= 6'd0;
            s9_M     <= 16'sd0;
        end else begin
            s9_valid <= s8_valid;
            s9_r     <= s8_r;
            s9_r3    <= s8_r3;
            s9_r4    <= s8_r4;
            s9_t2    <= mul_t2;
            s9_J     <= s8_J;
            s9_M     <= s8_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 10: t3 = A3_64 * r3
    // -------------------------------------------------------------------------
    reg        s10_valid;
    reg [31:0] s10_r;
    reg [31:0] s10_r4;
    reg [31:0] s10_t2;
    reg [31:0] s10_t3;
    reg [5:0]  s10_J;
    reg signed [15:0] s10_M;

    wire [31:0] mul_t3;
    fp32_mul u_mul_t3 (
        .a (A3_64),
        .b (s9_r3),
        .y (mul_t3)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s10_valid <= 1'b0;
            s10_r     <= 32'h0;
            s10_r4    <= 32'h0;
            s10_t2    <= 32'h0;
            s10_t3    <= 32'h0;
            s10_J     <= 6'd0;
            s10_M     <= 16'sd0;
        end else begin
            s10_valid <= s9_valid;
            s10_r     <= s9_r;
            s10_r4    <= s9_r4;
            s10_t2    <= s9_t2;
            s10_t3    <= mul_t3;
            s10_J     <= s9_J;
            s10_M     <= s9_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 11: t4 = A4_64 * r4
    // -------------------------------------------------------------------------
    reg        s11_valid;
    reg [31:0] s11_r;
    reg [31:0] s11_t2;
    reg [31:0] s11_t3;
    reg [31:0] s11_t4;
    reg [5:0]  s11_J;
    reg signed [15:0] s11_M;

    wire [31:0] mul_t4;
    fp32_mul u_mul_t4 (
        .a (A4_64),
        .b (s10_r4),
        .y (mul_t4)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s11_valid <= 1'b0;
            s11_r     <= 32'h0;
            s11_t2    <= 32'h0;
            s11_t3    <= 32'h0;
            s11_t4    <= 32'h0;
            s11_J     <= 6'd0;
            s11_M     <= 16'sd0;
        end else begin
            s11_valid <= s10_valid;
            s11_r     <= s10_r;
            s11_t2    <= s10_t2;
            s11_t3    <= s10_t3;
            s11_t4    <= mul_t4;
            s11_J     <= s10_J;
            s11_M     <= s10_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 12: sum23 = t2 + t3
    // -------------------------------------------------------------------------
    reg        s12_valid;
    reg [31:0] s12_r;
    reg [31:0] s12_t4;
    reg [31:0] s12_sum23;
    reg [5:0]  s12_J;
    reg signed [15:0] s12_M;

    wire [31:0] add_23;
    fp32_add u_add_23 (
        .a (s11_t2),
        .b (s11_t3),
        .y (add_23)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s12_valid <= 1'b0;
            s12_r     <= 32'h0;
            s12_t4    <= 32'h0;
            s12_sum23 <= 32'h0;
            s12_J     <= 6'd0;
            s12_M     <= 16'sd0;
        end else begin
            s12_valid <= s11_valid;
            s12_r     <= s11_r;
            s12_t4    <= s11_t4;
            s12_sum23 <= add_23;
            s12_J     <= s11_J;
            s12_M     <= s11_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 13: sum234 = sum23 + t4
    // -------------------------------------------------------------------------
    reg        s13_valid;
    reg [31:0] s13_r;
    reg [31:0] s13_sum234;
    reg [5:0]  s13_J;
    reg signed [15:0] s13_M;

    wire [31:0] add_234;
    fp32_add u_add_234 (
        .a (s12_sum23),
        .b (s12_t4),
        .y (add_234)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s13_valid  <= 1'b0;
            s13_r      <= 32'h0;
            s13_sum234 <= 32'h0;
            s13_J      <= 6'd0;
            s13_M      <= 16'sd0;
        end else begin
            s13_valid  <= s12_valid;
            s13_r      <= s12_r;
            s13_sum234 <= add_234;
            s13_J      <= s12_J;
            s13_M      <= s12_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 14: p = r + sum234
    // -------------------------------------------------------------------------
    reg        s14_valid;
    reg [31:0] s14_p;
    reg [5:0]  s14_J;
    reg signed [15:0] s14_M;

    wire [31:0] add_p;
    fp32_add u_add_p (
        .a (s13_r),
        .b (s13_sum234),
        .y (add_p)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s14_valid <= 1'b0;
            s14_p     <= 32'h0;
            s14_J     <= 6'd0;
            s14_M     <= 16'sd0;
        end else begin
            s14_valid <= s13_valid;
            s14_p     <= add_p;
            s14_J     <= s13_J;
            s14_M     <= s13_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 15: one_plus_p = 1.0 + p
    // -------------------------------------------------------------------------
    reg        s15_valid;
    reg [31:0] s15_one_plus_p;
    reg [5:0]  s15_J;
    reg signed [15:0] s15_M;

    wire [31:0] add_one;
    fp32_add u_add_one (
        .a (FP32_ONE),
        .b (s14_p),
        .y (add_one)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s15_valid     <= 1'b0;
            s15_one_plus_p <= 32'h0;
            s15_J         <= 6'd0;
            s15_M         <= 16'sd0;
        end else begin
            s15_valid     <= s14_valid;
            s15_one_plus_p <= add_one;
            s15_J         <= s14_J;
            s15_M         <= s14_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 16: core = T[J] * one_plus_p
    // -------------------------------------------------------------------------
    reg        s16_valid;
    reg [31:0] s16_core;
    reg signed [15:0] s16_M;

    wire [31:0] T_J = exp2_table_64[s15_J];

    wire [31:0] mul_core;
    fp32_mul u_mul_core (
        .a (T_J),
        .b (s15_one_plus_p),
        .y (mul_core)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s16_valid <= 1'b0;
            s16_core  <= 32'h0;
            s16_M     <= 16'sd0;
        end else begin
            s16_valid <= s15_valid;
            s16_core  <= mul_core;
            s16_M     <= s15_M;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 17: Scale by 2^M via exponent adjustment
    // -------------------------------------------------------------------------
    reg        s17_valid;
    reg [31:0] s17_y;

    wire        sign_core  = s16_core[31];
    wire [7:0]  exp_core   = s16_core[30:23];
    wire [22:0] frac_core  = s16_core[22:0];

    wire signed [9:0] E_core  = $signed({1'b0, exp_core}) - 10'sd127;
    wire signed [9:0] s16_M_10bit = $signed(s16_M[9:0]);
    wire signed [9:0] E_final = E_core + s16_M_10bit;

    reg [31:0] y_bits;

    always @(*) begin
        if (E_final > 10'sd127) begin
            // Overflow -> +INF
            y_bits = {sign_core, 8'hFF, 23'h0};
        end else if (E_final < -10'sd126) begin
            // Underflow -> 0
            y_bits = {sign_core, 8'h00, 23'h0};
        end else begin
            // Normal case
            y_bits = {sign_core, (E_final[7:0] + 8'd127), frac_core};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s17_valid <= 1'b0;
            s17_y     <= 32'h0;
        end else begin
            s17_valid <= s16_valid;
            s17_y     <= y_bits;
        end
    end

    // -------------------------------------------------------------------------
    // Stage 18: Output register
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_y     <= 32'h0;
        end else begin
            out_valid <= s17_valid;
            out_y     <= s17_y;
        end
    end

endmodule
