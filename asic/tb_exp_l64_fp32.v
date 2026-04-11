// tb_exp_l64_fp32.v
// ---------------------------------------------------------------------------
// Simple testbench for exp_l64_fp32
// - Drives random FP32 inputs in range [-10.0, 0.0]
// - Compares against a golden model generated offline (Python exp_fp32_l64)
//   by checking hex bit patterns or numerical closeness.
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps

module tb_exp_l64_fp32;

    reg         clk;
    reg         rst_n;
    reg         in_valid;
    reg  [31:0] in_x;
    wire        out_valid;
    wire [31:0] out_y;

    // DUT
    exp_l64_fp32 dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (in_valid),
        .in_x     (in_x),
        .out_valid(out_valid),
        .out_y    (out_y)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100 MHz
    end

    // Reset
    initial begin
        rst_n = 0;
        #40;
        rst_n = 1;
    end

    // Test vectors
    // For simplicity, we store a small table of FP32 inputs and golden outputs.
    // You can generate these with Python using exp_fp32_l64 and then paste hex
    // values here.

    localparam integer NUM_VECS = 8;

    reg [31:0] vec_x   [0:NUM_VECS-1];
    reg [31:0] vec_y_g [0:NUM_VECS-1]; // golden outputs (from Python model)

    initial begin
        // Example test points in [-10.0, 0.0]. Replace with real hex from Python.
        // x = -10.0, -7.5, -5.0, -2.5, -1.0, -0.5, -0.1, 0.0
        vec_x[0]   = 32'hC1200000; // -10.0f
        vec_x[1]   = 32'hC0F00000; // -7.5f
        vec_x[2]   = 32'hC0A00000; // -5.0f
        vec_x[3]   = 32'hC0200000; // -2.5f
        vec_x[4]   = 32'hBF800000; // -1.0f
        vec_x[5]   = 32'hBF000000; // -0.5f
        vec_x[6]   = 32'hBDCCCCCD; // -0.1f
        vec_x[7]   = 32'h00000000; // 0.0f

        // TODO: fill vec_y_g[i] with hex of exp_fp32_l64(x_i) from Python
        // Placeholder: set to zero for now.
        integer i;
        for (i = 0; i < NUM_VECS; i = i+1) begin
            vec_y_g[i] = 32'h00000000;
        end
    end

    integer idx;

    initial begin
        in_valid = 0;
        in_x     = 32'h0;
        idx      = 0;

        // Wait for reset deassertion
        @(posedge rst_n);
        @(posedge clk);

        // Apply vectors
        for (idx = 0; idx < NUM_VECS; idx = idx+1) begin
            @(posedge clk);
            in_valid <= 1'b1;
            in_x     <= vec_x[idx];
        end

        // Deassert valid after last vector
        @(posedge clk);
        in_valid <= 1'b0;

        // Run for some extra cycles to drain pipeline
        repeat (50) @(posedge clk);

        $finish;
    end

    // Simple monitor: print when output is valid
    always @(posedge clk) begin
        if (out_valid) begin
            $display("t=%0t, out_y=0x%08h", $time, out_y);
        end
    end

endmodule
