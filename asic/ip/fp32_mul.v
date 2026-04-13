// fp32_mul.v
// ---------------------------------------------------------------------------
// IEEE-754 single-precision FP32 multiplier.
// Combinational, Verilog-2001, Yosys-synthesizable.
//
// Ports:
//   a [31:0]  – FP32 operand A
//   b [31:0]  – FP32 operand B
//   y [31:0]  – FP32 result = a * b
//
// Policy:
//   • Flush-to-Zero (FTZ): denormal inputs treated as ±0;
//     denormal outputs flushed to ±0.
//   • Round-to-Nearest-Even (RNE).
//   • NaN payload: output quiet NaN = 32'hFFFF_FFFF.
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps

module fp32_mul (
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
    wire a_is_zero = (a_exp == 8'h00);  // FTZ: denormal treated as zero
    wire b_is_zero = (b_exp == 8'h00);  // FTZ: denormal treated as zero

    // Restore implicit leading 1 (normals only; denormals are flushed)
    wire [23:0] a_sig = a_is_zero ? 24'h0 : {1'b1, a_mant};
    wire [23:0] b_sig = b_is_zero ? 24'h0 : {1'b1, b_mant};

    // -----------------------------------------------------------------------
    // Step 2: Multiply
    // -----------------------------------------------------------------------
    wire        prod_sign = a_sign ^ b_sign;

    // 9-bit signed exponent sum with de-bias: (a_exp + b_exp - 127)
    // Using 9 bits to detect overflow/underflow.
    wire [8:0]  prod_exp_raw = {1'b0, a_exp} + {1'b0, b_exp} - 9'd127;

    // 24x24 unsigned significand product → 48 bits
    // Yosys infers a multiplier array automatically.
    wire [47:0] prod_sig = a_sig * b_sig;

    // -----------------------------------------------------------------------
    // Step 3: Normalise
    //   IEEE 754 normal × normal product has leading 1 at bit 47 or bit 46.
    //   • bit 47 == 1: product is in range [2, 4) — shift right 1, exp+1
    //   • bit 46 == 1: product is in range [1, 2) — already normal
    //   (For zero inputs, prod_sig == 0; handled by special-case mux.)
    // -----------------------------------------------------------------------
    wire prod_overflow_bit = prod_sig[47];  // 1 → shift right needed

    // Guard (G), Round (R), Sticky (S) bits from the 48-bit product.
    // After deciding the shift, mantissa = bits [46:24] (23 bits).
    //   no shift: mantissa = prod_sig[46:24], G=prod_sig[23], R=prod_sig[22], S=|prod_sig[21:0]
    //   shift  1: mantissa = prod_sig[47:25], G=prod_sig[24], R=prod_sig[23], S=|prod_sig[22:0]

    wire [22:0] norm_mant = prod_overflow_bit ? prod_sig[46:24] : prod_sig[45:23];
    wire        G         = prod_overflow_bit ? prod_sig[23]    : prod_sig[22];
    wire        R_bit     = prod_overflow_bit ? prod_sig[22]    : prod_sig[21];
    wire        S_bit     = prod_overflow_bit ? |prod_sig[21:0] : |prod_sig[20:0];

    // Exponent adjustment for normalisation shift
    wire [8:0]  norm_exp  = prod_exp_raw + {8'b0, prod_overflow_bit};

    // -----------------------------------------------------------------------
    // Step 4: Round-to-Nearest-Even
    // -----------------------------------------------------------------------
    wire round_up = G & (R_bit | S_bit | norm_mant[0]);

    wire [23:0] rounded    = {1'b0, norm_mant} + {23'b0, round_up};
    wire [22:0] round_mant = rounded[22:0];
    wire        mant_carry = rounded[23];

    wire [8:0]  final_exp  = norm_exp + {8'b0, mant_carry};
    wire [22:0] final_mant = mant_carry ? 23'h0 : round_mant;

    // -----------------------------------------------------------------------
    // Step 5: Special-case output mux (priority order)
    // -----------------------------------------------------------------------
    wire any_nan     = a_is_nan | b_is_nan;
    wire inf_times_0 = (a_is_inf & b_is_zero) | (a_is_zero & b_is_inf);
    wire any_inf     = a_is_inf | b_is_inf;
    wire any_zero    = a_is_zero | b_is_zero;

    // Overflow: final_exp >= 255 (and neither input was ±0)
    wire overflow  = (final_exp >= 9'd255) && !any_zero;
    // Underflow: exponent went to 0 or negative → FTZ
    // final_exp[8] is the sign bit of the 9-bit value (set when negative)
    wire underflow = (final_exp[8] || final_exp == 9'd0) && !any_zero;

    assign y =
        any_nan     ? 32'hFFFF_FFFF                       : // NaN propagate
        inf_times_0 ? 32'hFFFF_FFFF                       : // Inf × 0 → NaN
        any_inf     ? {prod_sign, 8'hFF, 23'h0}           : // ±Inf
        any_zero    ? {prod_sign, 31'b0}                  : // ±0
        overflow    ? {prod_sign, 8'hFF, 23'h0}           : // overflow → ±Inf
        underflow   ? {prod_sign, 31'b0}                  : // FTZ → ±0
                      {prod_sign, final_exp[7:0], final_mant}; // normal

endmodule
