# Makefile for FP32 IP cores
# Run from: asic/ip/
# Targets: sim_all, synth_all, sim_<module>, synth_<module>, vectors, clean
#
# Tools required: iverilog, vvp, yosys, python3

IVERILOG  = iverilog
VVP       = vvp
YOSYS     = yosys
PYTHON    = python3

IP_DIR    = $(CURDIR)
TB_DIR    = $(CURDIR)/tb

IFLAGS    = -g2001 -Wall

# Source files
SRC_ADD   = $(IP_DIR)/fp32_add.v
SRC_MUL   = $(IP_DIR)/fp32_mul.v
SRC_SUB   = $(IP_DIR)/fp32_add.v $(IP_DIR)/fp32_sub.v
SRC_F2I   = $(IP_DIR)/fp32_to_int.v
SRC_I2F   = $(IP_DIR)/int_to_fp32.v

# ---------------------------------------------------------------------------
# Simulation targets
# Vector hex files are placed in TB_DIR; vvp is run from TB_DIR so that
# $fopen("vec_*.hex") resolves correctly.
# ---------------------------------------------------------------------------

.PHONY: sim_all sim_fp32_add sim_fp32_sub sim_fp32_mul sim_fp32_to_int sim_int_to_fp32 sim_roundtrip

sim_all: sim_fp32_add sim_fp32_sub sim_fp32_mul sim_fp32_to_int sim_int_to_fp32 sim_roundtrip

sim_fp32_add: vectors
	$(IVERILOG) $(IFLAGS) -o $(TB_DIR)/sim_fp32_add.vvp \
	    $(SRC_ADD) $(TB_DIR)/tb_fp32_add.v
	cd $(TB_DIR) && $(VVP) sim_fp32_add.vvp

sim_fp32_sub: vectors
	$(IVERILOG) $(IFLAGS) -o $(TB_DIR)/sim_fp32_sub.vvp \
	    $(SRC_SUB) $(TB_DIR)/tb_fp32_sub.v
	cd $(TB_DIR) && $(VVP) sim_fp32_sub.vvp

sim_fp32_mul: vectors
	$(IVERILOG) $(IFLAGS) -o $(TB_DIR)/sim_fp32_mul.vvp \
	    $(SRC_MUL) $(TB_DIR)/tb_fp32_mul.v
	cd $(TB_DIR) && $(VVP) sim_fp32_mul.vvp

sim_fp32_to_int: vectors
	$(IVERILOG) $(IFLAGS) -o $(TB_DIR)/sim_fp32_to_int.vvp \
	    $(SRC_F2I) $(TB_DIR)/tb_fp32_to_int.v
	cd $(TB_DIR) && $(VVP) sim_fp32_to_int.vvp

sim_int_to_fp32: vectors
	$(IVERILOG) $(IFLAGS) -o $(TB_DIR)/sim_int_to_fp32.vvp \
	    $(SRC_I2F) $(TB_DIR)/tb_int_to_fp32.v
	cd $(TB_DIR) && $(VVP) sim_int_to_fp32.vvp

sim_roundtrip:
	$(IVERILOG) $(IFLAGS) -o $(TB_DIR)/sim_roundtrip.vvp \
	    $(SRC_I2F) $(SRC_F2I) $(TB_DIR)/tb_roundtrip.v
	cd $(TB_DIR) && $(VVP) sim_roundtrip.vvp

# ---------------------------------------------------------------------------
# Synthesis targets (output in IP_DIR)
# ---------------------------------------------------------------------------

.PHONY: synth_all synth_fp32_add synth_fp32_sub synth_fp32_mul synth_fp32_to_int synth_int_to_fp32

synth_all: synth_fp32_add synth_fp32_sub synth_fp32_mul synth_fp32_to_int synth_int_to_fp32

synth_fp32_add: $(SRC_ADD)
	$(YOSYS) -p "read_verilog $(SRC_ADD); \
	             synth -top fp32_add; \
	             write_json $(IP_DIR)/fp32_add.json" \
	         2>&1 | tee $(IP_DIR)/synth_fp32_add.log

synth_fp32_sub: $(SRC_SUB)
	$(YOSYS) -p "read_verilog $(SRC_SUB); \
	             synth -top fp32_sub; \
	             write_json $(IP_DIR)/fp32_sub.json" \
	         2>&1 | tee $(IP_DIR)/synth_fp32_sub.log

synth_fp32_mul: $(SRC_MUL)
	$(YOSYS) -p "read_verilog $(SRC_MUL); \
	             synth -top fp32_mul; \
	             write_json $(IP_DIR)/fp32_mul.json" \
	         2>&1 | tee $(IP_DIR)/synth_fp32_mul.log

synth_fp32_to_int: $(SRC_F2I)
	$(YOSYS) -p "read_verilog $(SRC_F2I); \
	             synth -top fp32_to_int; \
	             write_json $(IP_DIR)/fp32_to_int.json" \
	         2>&1 | tee $(IP_DIR)/synth_fp32_to_int.log

synth_int_to_fp32: $(SRC_I2F)
	$(YOSYS) -p "read_verilog $(SRC_I2F); \
	             synth -top int_to_fp32; \
	             write_json $(IP_DIR)/int_to_fp32.json" \
	         2>&1 | tee $(IP_DIR)/synth_int_to_fp32.log

# ---------------------------------------------------------------------------
# Vector generation
# ---------------------------------------------------------------------------

.PHONY: vectors
vectors: $(TB_DIR)/vec_int_to_fp32.hex

$(TB_DIR)/vec_int_to_fp32.hex: $(TB_DIR)/gen_vectors.py
	cd $(TB_DIR) && $(PYTHON) gen_vectors.py

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

.PHONY: clean
clean:
	rm -f $(TB_DIR)/*.vvp $(TB_DIR)/*.vcd
	rm -f $(IP_DIR)/*.json $(IP_DIR)/synth_*.log
	rm -f $(TB_DIR)/vec_*.hex
