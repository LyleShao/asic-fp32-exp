// tb_fp32_add.v  – self-checking testbench for fp32_add
`timescale 1ns/1ps
module tb_fp32_add;
    reg  [31:0] a, b;
    wire [31:0] y;
    integer i, errors, fd;
    reg [31:0] a_vec, b_vec, y_exp;

    fp32_add dut (.a(a), .b(b), .y(y));

    // NaN comparison: both exp==FF && mant!=0 is a match
    function check_nan;
        input [31:0] got, exp;
        begin
            if (((got[30:23] == 8'hFF) && (got[22:0] != 23'h0)) &&
                ((exp[30:23] == 8'hFF) && (exp[22:0] != 23'h0)))
                check_nan = 1'b1;  // both NaN → pass
            else
                check_nan = (got === exp);
        end
    endfunction

    initial begin
        errors = 0;
        fd = $fopen("vec_fp32_add.hex", "r");
        if (fd == 0) begin $display("ERROR: cannot open vec_fp32_add.hex"); $finish; end

        i = 0;
        while (!$feof(fd)) begin
            if ($fscanf(fd, "%h %h %h\n", a_vec, b_vec, y_exp) == 3) begin
                a = a_vec; b = b_vec;
                #1;
                if (!check_nan(y, y_exp)) begin
                    $display("FAIL [%0d]: a=32'h%08h b=32'h%08h  got=32'h%08h  exp=32'h%08h",
                             i, a, b, y, y_exp);
                    errors = errors + 1;
                end
                i = i + 1;
            end
        end
        $fclose(fd);
        if (errors == 0) $display("PASS: fp32_add — %0d vectors, 0 errors", i);
        else             $display("FAIL: fp32_add — %0d/%0d vectors failed", errors, i);
        $finish;
    end
endmodule
