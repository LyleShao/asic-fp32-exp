// int_to_fp32.v
// ---------------------------------------------------------------------------
// Convert signed 16-bit integer to IEEE-754 single-precision float (FP32).
// Combinational, Verilog-2001, Yosys-synthesizable.
//
// Port:
//   a  [15:0] signed  – input integer  (-32768 .. +32767)
//   y  [31:0]         – IEEE-754 FP32 result
//
// Notes:
//   • A signed 16-bit integer has at most 15 magnitude bits.  The FP32
//     mantissa field holds 23 bits (plus implicit leading 1 = 24 total).
//     15 < 24, so every integer value is representable exactly — no rounding.
//   • a == 0         → +0.0  (32'h0000_0000)
//   • a == -32768    → the bit pattern 0xC7000000 (-32768.0 in FP32)
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps

module int_to_fp32 (
    input  signed [15:0] a,
    output        [31:0] y
);

    // -----------------------------------------------------------------------
    // Step 1: sign and magnitude
    // -----------------------------------------------------------------------
    wire        sign_bit = a[15];
    // Two's complement magnitude.  For a == -32768, (~a+1) wraps to 32768
    // which is 16'h8000; the leading-one at bit 15 is handled by lzc = 0.
    wire [15:0] mag = sign_bit ? (~a + 1'b1) : a;

    // -----------------------------------------------------------------------
    // Step 2: leading-one count via casez priority encoder
    //   lzc = number of leading zeros in mag[15:0]
    //   (i.e., position of highest set bit is 15-lzc)
    // -----------------------------------------------------------------------
    reg [3:0] lzc;

    always @(*) begin
        casez (mag[15:0])
            16'b1???_????_????_????: lzc = 4'd0;
            16'b01??_????_????_????: lzc = 4'd1;
            16'b001?_????_????_????: lzc = 4'd2;
            16'b0001_????_????_????: lzc = 4'd3;
            16'b0000_1???_????_????: lzc = 4'd4;
            16'b0000_01??_????_????: lzc = 4'd5;
            16'b0000_001?_????_????: lzc = 4'd6;
            16'b0000_0001_????_????: lzc = 4'd7;
            16'b0000_0000_1???_????: lzc = 4'd8;
            16'b0000_0000_01??_????: lzc = 4'd9;
            16'b0000_0000_001?_????: lzc = 4'd10;
            16'b0000_0000_0001_????: lzc = 4'd11;
            16'b0000_0000_0000_1???: lzc = 4'd12;
            16'b0000_0000_0000_01??: lzc = 4'd13;
            16'b0000_0000_0000_001?: lzc = 4'd14;
            16'b0000_0000_0000_0001: lzc = 4'd15;
            default:                 lzc = 4'd15; // mag == 0, handled below
        endcase
    end

    // -----------------------------------------------------------------------
    // Step 3: normalise
    //   Shift mag left by lzc so the leading 1 sits at bit 15.
    //   The mantissa stored in FP32 is the 15 bits below that leading 1,
    //   left-padded with 8 zeros to fill all 23 mantissa bits.
    // -----------------------------------------------------------------------
    wire [15:0] mant_shifted = mag << lzc;
    // mant_shifted[15] == 1 (the implicit leading 1, not stored).
    // Stored mantissa: bits [14:0] of mant_shifted, then 8 zero bits.
    wire [22:0] mantissa = {mant_shifted[14:0], 8'b0};

    // Biased exponent: value = 2^(15-lzc), so exponent = (15-lzc)+127 = 142-lzc
    wire [7:0] exponent = 8'd142 - {4'b0, lzc};

    // -----------------------------------------------------------------------
    // Step 4: output mux
    // -----------------------------------------------------------------------
    assign y = (a == 16'sd0) ? 32'h0000_0000
                             : {sign_bit, exponent, mantissa};

endmodule
