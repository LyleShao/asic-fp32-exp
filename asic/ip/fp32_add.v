// fp32_add.v
// ---------------------------------------------------------------------------
// IEEE-754 single-precision FP32 adder.
// Combinational, Verilog-2001, Yosys-synthesizable.
//
// Ports:
//   a [31:0]  – FP32 operand A
//   b [31:0]  – FP32 operand B
//   y [31:0]  – FP32 result = a + b
//
// Policy:
//   • Flush-to-Zero (FTZ): denormal inputs treated as ±0;
//     denormal outputs flushed to ±0.
//   • Round-to-Nearest-Even (RNE).
//   • NaN payload: output quiet NaN = 32'hFFFF_FFFF.
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps

module fp32_add (
    input  [31:0] a,
    input  [31:0] b,
    output [31:0] y
);

    // -----------------------------------------------------------------------
    // Step 1: Unpack
    // -----------------------------------------------------------------------
    wire        a_sign = a[31];
    wire [7:0]  a_exp  = a[30:23];
    wire [22:0] a_mant = a[22:0];

    wire        b_sign = b[31];
    wire [7:0]  b_exp  = b[30:23];
    wire [22:0] b_mant = b[22:0];

    // Special value detection
    wire a_is_nan  = (a_exp == 8'hFF) && (a_mant != 23'h0);
    wire b_is_nan  = (b_exp == 8'hFF) && (b_mant != 23'h0);
    wire a_is_inf  = (a_exp == 8'hFF) && (a_mant == 23'h0);
    wire b_is_inf  = (b_exp == 8'hFF) && (b_mant == 23'h0);
    wire a_is_zero = (a_exp == 8'h00);   // FTZ: denormal treated as zero
    wire b_is_zero = (b_exp == 8'h00);   // FTZ: denormal treated as zero

    // Restore implicit leading 1 (for normals only; denormals flushed)
    wire [23:0] a_sig = a_is_zero ? 24'h0 : {1'b1, a_mant};
    wire [23:0] b_sig = b_is_zero ? 24'h0 : {1'b1, b_mant};

    // -----------------------------------------------------------------------
    // Step 2: Compare magnitudes and swap so L (large) >= S (small)
    //   Compare by (exp, mant) — this gives |a| vs |b| correctly since both
    //   are normalised with the same exponent bias.
    // -----------------------------------------------------------------------
    wire a_mag_gte_b = {a_exp, a_mant} >= {b_exp, b_mant};

    wire        l_sign = a_mag_gte_b ? a_sign : b_sign;
    wire [7:0]  l_exp  = a_mag_gte_b ? a_exp  : b_exp;
    wire [23:0] l_sig  = a_mag_gte_b ? a_sig  : b_sig;

    wire        s_sign = a_mag_gte_b ? b_sign : a_sign;
    wire [7:0]  s_exp  = a_mag_gte_b ? b_exp  : a_exp;
    wire [23:0] s_sig  = a_mag_gte_b ? b_sig  : a_sig;

    // -----------------------------------------------------------------------
    // Step 3: Align smaller operand
    //   Shift s_sig right by exp_diff.  We need G, R, S bits for rounding.
    //   Work with a 27-bit extended significand:
    //     [26:3] = significand, [2]=G, [1]=R, [0]=S (sticky)
    //   Total shift range we care about: 0..26. Beyond 26, the sticky bit
    //   absorbs everything.
    // -----------------------------------------------------------------------
    wire [7:0] exp_diff_raw = l_exp - s_exp;
    // Cap shift at 27 (everything beyond is just sticky)
    wire [4:0] shift = (exp_diff_raw > 8'd27) ? 5'd27 : exp_diff_raw[4:0];

    // Build 50-bit extended small significand: {s_sig[23:0], 26'b0}
    // Then logically right-shift by `shift`, extracting GRS from the bottom.
    //
    // We implement this as:
    //   shifted[49:0] = {s_sig, 26'b0} >> shift
    //   l_ext[23:0]   = shifted[49:26]   (aligned significand, 24 bits)
    //   guard         = shifted[25]
    //   round         = shifted[24]
    //   sticky        = |shifted[23:0]
    //
    // To synthesize cleanly without a big barrel shifter on 50 bits,
    // use a 27-bit quantity (sig + 3 rounding bits) shifted right.
    // Any bits shifted past position 0 go into the sticky.

    wire [50:0] s_ext_full = {s_sig, 27'b0};    // 51 bits total
    wire [50:0] s_shifted  = s_ext_full >> shift; // right shift

    wire [23:0] s_aligned = s_shifted[50:27];
    wire        guard     = s_shifted[26];
    wire        rnd       = s_shifted[25];
    // Sticky: any bit shifted past the round position
    // Compute from the lower bits that shift removed
    // sticky = |(s_ext_full << (51 - shift)) — any remainder
    // Simpler: sticky = |(s_ext_full & ((51'b1 << shift) - 1) & ~((51'b1 << 2) - 1))
    // Easiest in synthesizable Verilog: compute separately
    // We need sticky = 1 if any bit below position `round` was shifted out.
    // Since s_ext_full = {s_sig, 27'b0}, bits [26:0] = 0 initially.
    // After right shift by `shift`:
    //   - The 27 appended zeros start at bit 26 and go to 0.
    //   - sticky collects bits [24:0] of s_shifted (everything below round).
    wire sticky_raw = |s_shifted[24:0];

    // If shift >= 27, the entire significand is shifted into sticky territory
    wire all_shifted = (exp_diff_raw > 8'd27);
    wire sticky = all_shifted ? (|s_sig) : sticky_raw;

    // -----------------------------------------------------------------------
    // Step 4: Add or subtract
    //   op_sub = 1 when effective operation is subtraction
    //   Since |l| >= |s|, subtraction cannot produce a borrow from the top.
    // -----------------------------------------------------------------------
    wire op_sub = l_sign ^ s_sign;

    // 25-bit adder: bit 24 = carry/borrow indicator
    wire [24:0] sum = op_sub ? ({1'b0, l_sig} - {1'b0, s_aligned})
                             : ({1'b0, l_sig} + {1'b0, s_aligned});

    // After subtraction the GRS bits negate conceptually, but since we
    // subtracted s_aligned from l, if s had any fractional remainder
    // (guard/round/sticky != 0) we need to borrow from sum LSB.
    // Implement: when op_sub and GRS != 0, reduce sum by 1 to make
    // the subtraction exact and set complemented GRS bits.
    //
    // Canonical approach: let sum represent the integer part of the
    // true difference.  The true fractional part is:
    //   frac = op_sub ? (2^3 - {guard, round, sticky}) : {guard, round, sticky}
    // with a borrow from sum[0] when op_sub && GRS != 0.
    //
    // We handle this by adjusting: if subtracting and GRS != 0, the
    // true mantissa sum is (sum - 1) in the integer part, and the
    // complemented GRS becomes (8 - GRS) = (~{guard,round,sticky}+1).
    // When subtracting and the smaller operand has a fractional part
    // (GRS != 0), the true integer difference is (sum - 1) and the
    // fractional part is (8 - GRS), i.e. the two's complement of GRS.
    wire grs_nonzero = guard | rnd | sticky;
    wire [24:0] sum_adj  = (op_sub && grs_nonzero) ? (sum - 25'h1) : sum;
    wire [2:0] frac_in  = {guard, rnd, sticky};
    wire [2:0] frac_neg = (~frac_in + 3'b001);   // two's complement
    wire [2:0] frac_sel = (op_sub && grs_nonzero) ? frac_neg : frac_in;
    wire G = frac_sel[2];
    wire R = frac_sel[1];
    wire S_bit = frac_sel[0];

    // -----------------------------------------------------------------------
    // Step 5: Normalize
    //   Case A: sum[24] == 1 → carry out; shift right 1, exp+1
    //   Case B: sum[23] == 1 → already normal at bit 23
    //   Case C: leading-zero count on sum[23:0]; shift left, exp-lzc
    //   Case D: sum == 0 → exact cancellation
    // -----------------------------------------------------------------------
    wire sum_carry  = sum_adj[24];
    wire sum_normal = sum_adj[23];

    // LZC on sum_adj[23:0] (for left-shift case)
    reg [4:0] sum_lzc;
    always @(*) begin
        casez (sum_adj[23:0])
            24'b1???_????_????_????_????_????: sum_lzc = 5'd0;
            24'b01??_????_????_????_????_????: sum_lzc = 5'd1;
            24'b001?_????_????_????_????_????: sum_lzc = 5'd2;
            24'b0001_????_????_????_????_????: sum_lzc = 5'd3;
            24'b0000_1???_????_????_????_????: sum_lzc = 5'd4;
            24'b0000_01??_????_????_????_????: sum_lzc = 5'd5;
            24'b0000_001?_????_????_????_????: sum_lzc = 5'd6;
            24'b0000_0001_????_????_????_????: sum_lzc = 5'd7;
            24'b0000_0000_1???_????_????_????: sum_lzc = 5'd8;
            24'b0000_0000_01??_????_????_????: sum_lzc = 5'd9;
            24'b0000_0000_001?_????_????_????: sum_lzc = 5'd10;
            24'b0000_0000_0001_????_????_????: sum_lzc = 5'd11;
            24'b0000_0000_0000_1???_????_????: sum_lzc = 5'd12;
            24'b0000_0000_0000_01??_????_????: sum_lzc = 5'd13;
            24'b0000_0000_0000_001?_????_????: sum_lzc = 5'd14;
            24'b0000_0000_0000_0001_????_????: sum_lzc = 5'd15;
            24'b0000_0000_0000_0000_1???_????: sum_lzc = 5'd16;
            24'b0000_0000_0000_0000_01??_????: sum_lzc = 5'd17;
            24'b0000_0000_0000_0000_001?_????: sum_lzc = 5'd18;
            24'b0000_0000_0000_0000_0001_????: sum_lzc = 5'd19;
            24'b0000_0000_0000_0000_0000_1???: sum_lzc = 5'd20;
            24'b0000_0000_0000_0000_0000_01??: sum_lzc = 5'd21;
            24'b0000_0000_0000_0000_0000_001?: sum_lzc = 5'd22;
            24'b0000_0000_0000_0000_0000_0001: sum_lzc = 5'd23;
            default:                           sum_lzc = 5'd24; // all zero
        endcase
    end

    // Build normalized mantissa and exponent (pre-rounding)
    reg [22:0] norm_mant;
    reg [8:0]  norm_exp;  // 9-bit to detect overflow/underflow
    reg        norm_G, norm_R, norm_S;
    reg [26:0] lshift_out;  // 23 mantissa + G + R + 2 sticky bits after left shift

    always @(*) begin
        lshift_out = 27'b0; // default to avoid latches
        if (sum_carry) begin
            // Shift right 1: bit[24] was the carry, [23:1] are mantissa,
            // [0] shifts into guard (old G shifts into round, old round→sticky)
            norm_mant = sum_adj[23:1];
            norm_exp  = {1'b0, l_exp} + 9'd1;
            norm_G    = sum_adj[0];
            norm_R    = G;
            norm_S    = R | S_bit;
        end else if (sum_normal) begin
            // Already normalised at bit 23
            norm_mant = sum_adj[22:0];
            norm_exp  = {1'b0, l_exp};
            norm_G    = G;
            norm_R    = R;
            norm_S    = S_bit;
        end else begin
            // Left-shift to normalise: propagate the existing GRS bits
            // through the shift using a 27-bit intermediate.
            // Layout: [26:4] = mantissa (23 bits), [3]=G, [2]=R, [1:0]=S
            lshift_out = {sum_adj[22:0], G, R, S_bit, 1'b0} << sum_lzc;
            norm_mant = lshift_out[26:4];
            norm_exp  = {1'b0, l_exp} - {4'b0, sum_lzc};
            norm_G    = lshift_out[3];
            norm_R    = lshift_out[2];
            norm_S    = |lshift_out[1:0];
        end
    end

    // -----------------------------------------------------------------------
    // Step 6: Round-to-Nearest-Even
    // -----------------------------------------------------------------------
    wire round_up = norm_G & (norm_R | norm_S | norm_mant[0]);

    wire [23:0] rounded   = {1'b0, norm_mant} + {23'b0, round_up};
    wire [22:0] round_mant = rounded[22:0];
    wire        mant_carry = rounded[23];

    wire [8:0] final_exp  = norm_exp + {8'b0, mant_carry};
    wire [22:0] final_mant = mant_carry ? 23'h0 : round_mant;

    // -----------------------------------------------------------------------
    // Step 7: Determine result sign
    //   For exact cancellation (x + (-x)): IEEE RNE → +0.
    //   For (-0) + (-0): both same-sign zero → -0.
    //   Otherwise: sign of the larger-magnitude operand.
    // -----------------------------------------------------------------------
    wire result_zero = (sum_adj[24:0] == 25'h0) && !grs_nonzero;
    // When both inputs are negative zero (or both-negative and cancel),
    // the zero result carries a negative sign per IEEE 754.
    wire result_sign = result_zero ? (a_sign & b_sign) : l_sign;

    // -----------------------------------------------------------------------
    // Step 8: Special-case output mux (priority order)
    // -----------------------------------------------------------------------
    wire any_nan    = a_is_nan | b_is_nan;
    wire inf_cancel = a_is_inf & b_is_inf & (a_sign ^ b_sign);
    wire any_inf    = a_is_inf | b_is_inf;
    wire inf_sign   = a_is_inf ? a_sign : b_sign;  // sign of the Inf operand

    // Overflow: final_exp >= 255 (but not when the result is actually zero)
    wire overflow   = (final_exp >= 9'd255) && !result_zero;
    // Underflow: final_exp <= 0 (after rounding, exponent wrapped/underflowed)
    wire underflow  = (final_exp == 9'd0) && !result_zero;

    assign y =
        any_nan    ? 32'hFFFF_FFFF                        : // quiet NaN
        inf_cancel ? 32'hFFFF_FFFF                        : // Inf - Inf → NaN
        any_inf    ? {inf_sign, 8'hFF, 23'h0}             : // ±Inf
        overflow   ? {result_sign, 8'hFF, 23'h0}          : // overflow → ±Inf
        underflow  ? {result_sign, 31'b0}                 : // FTZ → ±0
        result_zero? {result_sign, 31'b0}                  : // zero → ±0
                     {result_sign, final_exp[7:0], final_mant}; // normal

endmodule
