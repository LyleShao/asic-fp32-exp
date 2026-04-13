// tb_roundtrip.v
// Exhaustive round-trip test: all 65536 signed 16-bit integers
// through int_to_fp32 then fp32_to_int must recover the original value.
`timescale 1ns/1ps
module tb_roundtrip;
    reg  signed [15:0] a_int;
    wire        [31:0] fp;
    wire signed [15:0] a_back;
    integer i, errors;

    int_to_fp32  u_i2f (.a(a_int), .y(fp));
    fp32_to_int  u_f2i (.a(fp),    .y(a_back));

    initial begin
        errors = 0;
        for (i = -32768; i <= 32767; i = i + 1) begin
            a_int = i[15:0];
            #1;
            if (a_back !== a_int) begin
                $display("FAIL: int=%0d → fp=32'h%08h → back=%0d", a_int, fp, a_back);
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("PASS: roundtrip — 65536 vectors, 0 errors");
        else             $display("FAIL: roundtrip — %0d/65536 failed", errors);
        $finish;
    end
endmodule
