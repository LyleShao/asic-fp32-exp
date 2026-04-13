// fp32_to_int.v
// ---------------------------------------------------------------------------
// Convert IEEE-754 single-precision float (FP32) to signed 16-bit integer.
// Combinational, Verilog-2001, Yosys-synthesizable.
//
// Port:
//   a  [31:0]         – IEEE-754 FP32 input
//   y  [15:0] signed  – signed 16-bit integer result
//
// Rounding: Round-to-Nearest-Even (RNE).
// Saturation:
//   • Values > +32767 saturate to +32767 (16'h7FFF)
//   • Values < -32768 saturate to -32768 (16'h8000)
//   • The exact value -32768.0 (0xC7000000) converts to -32768 (16'h8000)
//   • NaN, ±Inf → 0  (common embedded convention)
//
// FTZ policy: denormal inputs (exp == 0) treated as ±0 → output 0.
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps

module fp32_to_int (
    input        [31:0] a,
    output signed [15:0] y
);

    // -----------------------------------------------------------------------
    // Step 1: Unpack
    // -----------------------------------------------------------------------
    wire        a_sign = a[31];
    wire [7:0]  a_exp  = a[30:23];
    wire [22:0] a_mant = a[22:0];

    // Special value flags
    wire a_is_nan  = (a_exp == 8'hFF) && (a_mant != 23'h0);
    wire a_is_inf  = (a_exp == 8'hFF) && (a_mant == 23'h0);
    wire a_is_zero = (a_exp == 8'h00);  // includes +0, -0, FTZ denormals

    // Restore implicit leading 1 (for normals)
    wire [23:0] sig = {1'b1, a_mant};

    // -----------------------------------------------------------------------
    // Step 2: Compute true exponent (biased exponent - 127)
    //   9-bit to detect sign and overflow.
    //   true_exp < 0  → |value| < 1.0
    //   true_exp ≥ 15 → |value| ≥ 32768
    //   true_exp ∈ [0,14] → normal conversion range
    // -----------------------------------------------------------------------
    wire [8:0] true_exp = {1'b0, a_exp} - 9'd127;
    wire       exp_neg  = true_exp[8];  // 1 when true_exp < 0

    // -----------------------------------------------------------------------
    // Step 3: Extract integer magnitude and GRS bits.
    //
    //   The 24-bit significand sig = 1.mmm...mmm (bit 23 = implicit 1).
    //   For true_exp = N:
    //     integer magnitude = sig[23 : 23-N]   (N+1 bits; max 15 bits at N=14)
    //     G = sig[22-N]
    //     R = sig[21-N]
    //     S = |sig[20-N:0]
    //
    //   Strategy: build a 47-bit value {sig, 23'b0} and right-shift by
    //   (23 - true_exp).  After the shift:
    //     int_part  = work[46:31]  (16 bits, upper bits zero for small N)
    //     G         = work[30]
    //     R         = work[29]
    //     S         = |work[28:0]
    //
    //   This works because sig bit 23 lands at position 46-(23-N) = 23+N,
    //   and for N=14 that is position 37; int_part is bits [46:31], so
    //   bits [46:38] are zero and bits [37:31] hold the 7-bit MSBs — wait,
    //   still not right.
    //
    //   SIMPLER: right-shift sig by (23 - true_exp) → 24-bit quotient.
    //   The integer magnitude occupies bits [N:0] of the 24-bit shifted value.
    //   Maximum N=14, so int_part fits in bits [14:0], i.e. 15 bits ≤ 16'hFFFF.
    //   Use the bottom of sig to compute sticky via a separate left-shift.
    //
    //   rshift = 23 - true_exp (clamped to 0..23 for normal range).
    //   int_part   = sig >> rshift              → 24-bit (≤ 15 significant bits)
    //   shifted_lo = sig << (24 - rshift)       → 24-bit, holds G/R/S region
    //     G   = shifted_lo[23]
    //     R   = shifted_lo[22]
    //     S   = |shifted_lo[21:0]
    // -----------------------------------------------------------------------
    wire [4:0] rshift = (exp_neg || (true_exp >= 9'd23)) ? 5'd23
                                                         : (5'd23 - true_exp[4:0]);

    // Right-shift sig to get integer magnitude (24 → 16-bit result)
    wire [23:0] sig_shifted = sig >> rshift;
    wire [15:0] int_part    = sig_shifted[15:0];  // only ≤15 bits significant

    // Left-shift sig by (24 - rshift) to bring fractional bits up for GRS.
    // (24 - rshift) is in range [1..24] for rshift in [0..23].
    // Use 5-bit arithmetic; when rshift=0 lshift=24 (full shift out = sticky=0).
    wire [4:0]  lshift      = 5'd24 - rshift;     // 24 - rshift
    wire [23:0] sig_lshift  = (rshift == 5'd0) ? 24'b0 : (sig << lshift);
    wire        G           = sig_lshift[23];
    wire        R_bit       = sig_lshift[22];
    wire        S_bit       = |sig_lshift[21:0];

    // -----------------------------------------------------------------------
    // Step 4: Round-to-Nearest-Even
    // -----------------------------------------------------------------------
    wire round_up = G & (R_bit | S_bit | int_part[0]);

    wire [16:0] rounded_mag = {1'b0, int_part} + {16'b0, round_up};

    // -----------------------------------------------------------------------
    // Step 5: Special-case mux (priority order)
    // -----------------------------------------------------------------------
    // Exactly -32768.0 in FP32: 0xC7000000
    wire is_neg32768 = (a == 32'hC700_0000);

    // Overflow: true exponent ≥ 15 means |value| ≥ 32768
    wire exp_overflow = (!exp_neg) && (true_exp >= 9'd15);

    wire mag_overflows_pos = (rounded_mag > 17'd32767);
    wire mag_overflows_neg = (rounded_mag > 17'd32768);

    reg signed [15:0] result;

    always @(*) begin
        if (a_is_nan || a_is_inf || a_is_zero) begin
            result = 16'sd0;
        end else if (exp_neg) begin
            // |value| < 1.0 → rounds to 0 (or ±1 when exactly ±0.5)
            result = a_sign ? -($signed({15'b0, round_up}))
                            :   $signed({15'b0, round_up});
        end else if (is_neg32768) begin
            result = -16'sd32768;        // 16'sh8000
        end else if (exp_overflow) begin
            result = a_sign ? -16'sd32768 : 16'sd32767;
        end else begin
            // Normal conversion
            if (a_sign) begin
                if (mag_overflows_neg)
                    result = -16'sd32768;
                else
                    result = -($signed({1'b0, rounded_mag[15:0]}));
            end else begin
                if (mag_overflows_pos)
                    result = 16'sd32767;
                else
                    result = $signed({1'b0, rounded_mag[15:0]});
            end
        end
    end

    assign y = result;

endmodule
