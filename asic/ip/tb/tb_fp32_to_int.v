// tb_fp32_to_int.v
// Self-checking testbench for fp32_to_int.
// Loads vectors from vec_fp32_to_int.hex.
// For NaN outputs the expected value is 0 (per our convention).
`timescale 1ns/1ps

module tb_fp32_to_int;

    reg        [31:0] a;
    wire signed [15:0] y;
    integer i, errors, fd;
    reg [31:0] a_vec;
    reg [15:0] y_exp_raw;
    wire signed [15:0] y_exp;

    fp32_to_int dut (.a(a), .y(y));

    assign y_exp = $signed(y_exp_raw);

    initial begin
        errors = 0;
        fd = $fopen("vec_fp32_to_int.hex", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec_fp32_to_int.hex");
            $finish;
        end

        i = 0;
        while (!$feof(fd)) begin
            if ($fscanf(fd, "%h %h\n", a_vec, y_exp_raw) == 2) begin
                a = a_vec;
                #1;
                if (y !== y_exp) begin
                    $display("FAIL [%0d]: a=32'h%08h  got=%0d (16'h%04h)  exp=%0d (16'h%04h)",
                             i, a, y, y, y_exp, y_exp);
                    errors = errors + 1;
                end
                i = i + 1;
            end
        end

        $fclose(fd);
        if (errors == 0)
            $display("PASS: fp32_to_int — %0d vectors, 0 errors", i);
        else
            $display("FAIL: fp32_to_int — %0d/%0d vectors failed", errors, i);
        $finish;
    end
endmodule
