// tb_int_to_fp32.v
// Self-checking testbench: exhaustive 65536 vectors from vec_int_to_fp32.hex
`timescale 1ns/1ps

module tb_int_to_fp32;

    reg  signed [15:0] a;
    wire        [31:0] y;
    integer i, errors, fd;
    reg [15:0] a_vec;
    reg [31:0] y_exp;

    int_to_fp32 dut (.a(a), .y(y));

    initial begin
        errors = 0;
        fd = $fopen("vec_int_to_fp32.hex", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec_int_to_fp32.hex");
            $finish;
        end

        i = 0;
        while (!$feof(fd)) begin
            if ($fscanf(fd, "%h %h\n", a_vec, y_exp) == 2) begin
                a = $signed(a_vec);
                #1;
                if (y !== y_exp) begin
                    $display("FAIL [%0d]: a=%0d (16'h%04h)  got=32'h%08h  exp=32'h%08h",
                             i, a, a_vec, y, y_exp);
                    errors = errors + 1;
                end
                i = i + 1;
            end
        end

        $fclose(fd);
        if (errors == 0)
            $display("PASS: int_to_fp32 — %0d vectors, 0 errors", i);
        else
            $display("FAIL: int_to_fp32 — %0d/%0d vectors failed", errors, i);
        $finish;
    end
endmodule
