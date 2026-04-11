# ASIC-Oriented FP32 `exp(x)` for Safe Softmax

This repository contains **hardware-friendly implementations of `y = exp(x)`**
for IEEE-754 single precision (FP32), targeting ASIC datapaths with FP32
add/mul IP and small ROMs.

The design is inspired by and compatible with the table-driven algorithm in:

> Ping Tak Peter Tang, "Table-Driven Implementation of the Exponential
> Function in IEEE Floating-Point Arithmetic", ACM TOMS, 1989.

We provide two Python reference functions that can be translated almost
one-to-one into an RTL pipeline:

- `exp_fp32`   — Tang's original L=32 table-driven FP32 exp.
- `exp_fp32_l64` — an improved L=64 variant tuned for **safe-softmax**.

The rest of this document focuses on **`exp_fp32_l64`**, which we recommend for
new ASIC designs.

---

## 1. Quick Start (Safe Softmax)

```python
from exp import exp_fp32_l64

# Example: safe-softmax over a row x[] in FP32
row = [0.1, -3.2, 2.5, -1.0]
rowmax = max(row)

exps = [exp_fp32_l64(x - rowmax) for x in row]
Z = sum(exps)
softmax = [v / Z for v in exps]
```

Typical accuracy for `exp_fp32_l64` vs Python `math.exp`:

- On `x ∈ [-16, 16]`: max relative error ≈ **1e-6**.
- On `x ∈ [-64, 0]` (classic safe-softmax domain `xi - rowmax`):
  - max absolute error ≈ `7.6e-8`.
  - max relative error ≈ **4e-6** (20k random samples).

This is more than sufficient for numerically stable softmax in FP32.

---

## 2. Functions and Recommendation

All code lives in `exp.py`.

### 2.1 Overview

| Function        | Algorithm                | Range reduction | Poly degree | Table size | Accuracy (≈) on [-16,16] | Recommended use                     |
|-----------------|--------------------------|-----------------|------------|-----------:|--------------------------|--------------------------------------|
| `exp_fp32`      | Tang 1989, L=32          | L=32            | 3          |   32×2 FP32 | ~3.3e-6 rel vs `math.exp` | Reference to Tang (1989)            |
| `exp_fp32_l64`  | Improved, L=64 quartic   | L=64            | 4          |   64×1 FP32 | ~1.0e-6 rel vs `math.exp` | **Default for ASIC / safe-softmax** |

Both implementations:

- Operate in FP32 throughout (no extra precision required).
- Use small ROM tables and a handful of FP32 add/mul units.
- Fit comfortably under a budget of ≤16 FP32 multipliers and ≤2 KB ROM.

**Recommendation:**

- Use **`exp_fp32_l64`** for new ASIC designs, especially for safe-softmax.
- Keep `exp_fp32` as a Tang-compatible reference or for cross-checking.

### 2.2 `exp_fp32(x)` — Tang 1989, L = 32

```python
from exp import exp_fp32
```

- Implements Tang's original single-precision algorithm with:
  - Range reduction using `L = 32` subintervals.
  - Reduced argument represented as `R1 + R2` with a branch on `|N|`.
  - Degree-3 polynomial `p(r) = r + A1 r^2 + A2 r^3`.
  - 32-entry table of `2^(j/32)` stored as pairs `(S_lead, S_trail)`.
- Very faithful to the 1989 paper; slightly more complex control.

### 2.3 `exp_fp32_l64(x)` — Improved L = 64 Variant (Recommended)

```python
from exp import exp_fp32_l64
```

Key properties:

- Range reduction with **L = 64** subintervals.
- Simple, branch-free reduction and reconstruction logic.
- Fitted quartic polynomial for `exp(r) - 1` over a small range.
- Single FP32 table `EXP2_TABLE_64[j] = 2^(j/64)`.
- Excellent accuracy in the softmax domain.

This variant was designed to be easy to map to hardware: the Python code
essentially *is* an FP32 datapath with pipeline boundaries.

---

## 3. Algorithm (L = 64 Variant)

The `exp_fp32_l64` algorithm works as follows.

### 3.1 Special Cases

For inputs `x`:

- If `x` is NaN: return quiet NaN.
- If `x` is +∞: return +∞.
- If `x` is −∞: return +0.
- If `|x| > THRESHOLD_1` (≈ 341·ln2):
  - If `x > 0`: overflow → +∞.
  - If `x < 0`: underflow → +0.
- If `|x| < THRESHOLD_2` (≈ 2⁻²⁵): return `1 + x`.

These cases are handled in a front-end stage and bypass the main algorithm.

### 3.2 Range Reduction (L = 64)

For normal inputs, we first reduce `x` to a small argument `r`:

1. Compute integer `N`:

   ```text
   N = round_to_nearest_even( x * (64 / ln 2) )
   ```

2. Decompose `N` into:

   ```text
   J = N mod 64       // J in [0..63]
   M = (N - J) / 64   // integer
   ```

3. Compute reduced argument `r`:

   ```text
   r ≈ x - N * (ln 2 / 64)
   ```

4. With this reduction, `r` lies in a very small interval:

   ```text
   |r| <= ln(2) / 128 ≈ 0.0054
   ```

### 3.3 Polynomial Approximation

We approximate `exp(r) - 1` by a quartic polynomial:

```text
p(r) = r + A2 r^2 + A3 r^3 + A4 r^4
```

- `A2`, `A3`, `A4` are fitted (least squares) over `|r| <= ln(2)/128` and stored
  as FP32 constants in `exp.py`.
- All multiplications and additions are FP32 in the reference model (your ASIC
  can use the same or wider internal precision).

### 3.4 Table Lookup and Reconstruction

We precompute a small table:

```text
T[J] = 2^(J/64),    J = 0..63
```

stored as FP32 literals in `EXP2_TABLE_64`.

Given `r`, `p(r)`, `J`, `M` from the reduction step:

1. Compute:

   ```text
   core = T[J] * (1 + p(r))
   ```

2. Apply the `2^M` factor:

   ```text
   exp(x) ≈ 2^M * core
   ```

In hardware, the multiplication by `2^M` can be implemented by **adjusting the
exponent field** of `core` (IEEE-754-style `ldexp`), not a general FP
multiplication.

---

## 4. Mapping `exp_fp32_l64` to a Pipelined ASIC Datapath

Assuming FP32 IP blocks (`ADD`, `MUL`, `ABS`, comparisons, float↔int) and small
ROMs, you can implement `exp_fp32_l64` as a 5–6 stage pipeline.

### S0: Input / Special Cases

- Input: FP32 `x`.
- Operations:
  - Decode exponent/mantissa, detect NaN and ±Inf.
  - Compute `|x|` and compare with `THRESHOLD_1`, `THRESHOLD_2`.
  - Directly output NaN, +∞, +0, or `1+x` when applicable.
- Output to core pipeline:
  - Normalized FP32 `x`, valid flag.

### S1: Range Reduction — `N`, `J`, `M`

- Compute `tN = x * INV_L_64` (FP32 MUL).
- Convert `tN` to integer `N` via round-to-nearest-even (float→int).
- Integer logic:

  ```text
  J = N mod 64
  M = (N - J) / 64
  ```

  (Both very small blocks.)

- Register `x`, `J`, `M` for the next stage.

### S2: Reduced Argument `r`

- Convert `N` to FP32: `N_fp`.
- Compute `t = N_fp * LOG2_BY_64` (FP32 MUL).
- Compute `r = x - t` (FP32 ADD).
- Forward `J`, `M` with `r`.

### S3: Polynomial `p(r)`

- Compute powers:

  ```text
  r2 = r * r
  r3 = r2 * r
  r4 = r2 * r2
  ```

- Compute polynomial:

  ```text
  p = r + A2*r2 + A3*r3 + A4*r4
  ```

  using FP32 MUL+ADD blocks.

- Forward `J`, `M` with `p`.

> Note: You can choose how many multipliers to use in parallel. With up to
> ~6 MULs you can do this in 1–2 cycles; with fewer MULs you can reuse them
> across more cycles.

### S4: Table Lookup and `core`

- Use `J` to address ROM:

  ```text
  T[J] = EXP2_TABLE_64[J]
  ```

- Compute `one_plus_p = 1 + p` (FP32 ADD).
- Compute `core = T[J] * one_plus_p` (FP32 MUL).
- Forward `M` with `core`.

### S5: Scaling by `2^M` and FP32 Pack

- Interpret `core` as FP32 (`sign`, `exponent`, `mantissa`).
- Use integer arithmetic on the exponent field:

  ```text
  E_core  = core_exp - 127
  E_final = E_core + M
  ```

- Handle overflow/underflow on `E_final`.
- Repack sign/mantissa with bias-adjusted exponent (`E_final + 127`).
- Output final FP32 result.

This pipeline is essentially the same structure used in the Python implementation
of `exp_fp32_l64`, expressed in hardware terms.

---

## 5. ASIC Resource Comparison and Recommendation

Below are approximate upper bounds for a mostly parallel implementation using
FP32 operators. You can reduce multiplier count by sharing them across cycles.

### 5.1 `exp_fp32` (Tang L = 32)

- **ROM**
  - 32-entry table of `(S_lead, S_trail)` for `2^(j/32)`.
  - Size ≈ 32 × 2 × 32 bits = 2048 bits ≈ **256 bytes**.

- **FP32 multipliers (upper bound):** ~6–8
  - 1× `x * INV_L`.
  - 1–2× for `N1 * L1`, `N2 * L1` (range reduction path).
  - 2–3× for the degree-3 polynomial.
  - 1× for reconstruction `S * P`.

- **FP32 adders:** ~8–10
  - Forming `R1`, `R2`, `R`, polynomial accumulation, reconstruction.

- **Logic / control:**
  - Integer logic for `N`, `N1`, `N2`, `M = N1/32`.
  - Control around the `|N|` branch and combining `(S_lead, S_trail)`.

### 5.2 `exp_fp32_l64` (L = 64, quartic, recommended)

- **ROM**
  - 64-entry table `T[J] = 2^(J/64)` as FP32.
  - Size ≈ 64 × 32 bits = 2048 bits ≈ **256 bytes**.
  - A few FP32 constants: `INV_L_64`, `LOG2_BY_64`, `A2`, `A3`, `A4`.

- **FP32 multipliers (upper bound):** typically 6–9
  - 1× `x * INV_L_64`.
  - 1× `N * LOG2_BY_64`.
  - 3× for `r²`, `r³`, `r⁴`.
  - 3× for `A2*r²`, `A3*r³`, `A4*r⁴`.
  - 1× for `T[J] * (1 + p)`.
  - With scheduling, S3 can be implemented with fewer than 6 MULs.

- **FP32 adders:** ~8–10
  - `r = x - N*LOG2_BY_64`.
  - Polynomial sum.
  - `one_plus_p = 1 + p`.

- **Logic / control:**
  - Integer `N = round(x * INV_L_64)`, `J = N mod 64`, `M = (N - J)/64`.
  - Exponent-field adjustment for `2^M`.
  - No split tables or `|N|`-dependent reduction path.

### 5.3 Which is “more implementable”?

Both designs:

- Fit easily in a budget of **≤16 FP32 multipliers** and **≤2 KB ROM**.
- Have similar pipeline depth (≈5–6 stages).
- Use only standard FP32 operations and small ROM tables.

However:

- **`exp_fp32_l64`** has **simpler control and datapath**:
  - Single table `T[J]` instead of `(S_lead, S_trail)`.
  - No `|N|` branch in the reduction.
  - Cleaner S0–S5 pipeline.
- It also has **better accuracy** in the ranges we care about:
  - ~3.3e-6 rel error (Tang L=32) vs ~1.0e-6 (L=64) on [-16,16].
  - ~4e-6 rel error on [-64,0] for safe-softmax.

> **Recommendation:** For new ASIC designs, especially for safe-softmax, use
> **`exp_fp32_l64`** as the exp core. Keep `exp_fp32` as a Tang-compatible
> reference for validation and comparison.

---

## 6. Testing

### 6.1 Quick sanity test

From the project root:

```bash
python exp.py
```

This runs `test_exp()` (defined at the bottom of `exp.py`), which:

- Samples points in `[-100, 100]` (excluding overflow region).
- Reports max relative error for both:
  - `exp_fp32` (Tang L=32).
  - `exp_fp32_l64` (L=64 variant).
- Prints special-case behavior (NaN/Inf, etc.).

### 6.2 Focused test for safe-softmax

For safe-softmax we compute `exp(xi - rowmax)`, so the input to `exp` falls
into `x ∈ [-K, 0]` for some K (e.g. `K=64`). Example test:

```python
# test_exp_l64_softmax_range.py
import math, random
from exp import exp_fp32_l64

random.seed(2025)
N = 20000
min_x, max_x = -64.0, 0.0

max_abs = 0.0
max_rel = 0.0
max_abs_x = None
max_rel_x = None

for _ in range(N):
    x = random.uniform(min_x, max_x)
    y_true = math.exp(x)
    y_our  = exp_fp32_l64(x)

    abs_err = abs(y_true - y_our)
    if abs_err > max_abs:
        max_abs = abs_err
        max_abs_x = x

    rel_err = abs_err / y_true  # y_true > 0 here
    if rel_err > max_rel:
        max_rel = rel_err
        max_rel_x = x

print(f"N={N}")
print(f"x range: [{min_x}, {max_x}]")
print(f"max_abs_error={max_abs} at x={max_abs_x}")
print(f"max_rel_error={max_rel} at x={max_rel_x}")
```

### 6.3 Tang-style ULP tests

If you want to replicate Tang's ULP-style analysis:

- Restrict inputs to FP32 grid points.
- Compare against a high-precision reference:
  - `ref = round_to_fp32(math.exp(x32))`, where `x32` is FP32-rounded input.
- Measure ULP error between your implementation and `ref`.

The existing scripts `test_error.py`, `test_small.py` show how to compute ULP
errors for FP32.

---

## 8. Implementation Checklist (ASIC, FP32 IP)

This is a practical checklist for implementing `exp_fp32_l64` as an ASIC block
using FP32 add/mul/compare IP and a small ROM.

1. **Decide the numeric model**
   - Use FP32 throughout the datapath (recommended), or
   - Use FP32 at the boundaries and a slightly wider internal format if your IP
     supports it. In either case, keep the same algorithm and tables.

2. **Instantiate FP32 IP blocks**
   - At minimum:
     - FP32 adder/subtractor.
     - FP32 multiplier.
     - FP32 compare/abs (for thresholds and special cases).
     - FP32→integer and integer→FP32 converters (for N, J, M).

3. **Create ROMs / constant storage**
   - Scalar constants from `exp.py`:
     - `THRESHOLD_1`, `THRESHOLD_2`.
     - `INV_L_64`, `LOG2_BY_64`.
     - `A2_64`, `A3_64`, `A4_64`.
   - Table ROM:
     - `EXP2_TABLE_64[J] = 2^(J/64)`, 64 entries of FP32.

4. **Implement pipeline stages S0–S5**
   - S0: Input decode and special cases.
   - S1: `tN = x * INV_L_64`, `N = round_to_nearest_even(tN)`, compute `J` and `M`.
   - S2: Reduced argument `r = x - N*LOG2_BY_64`.
   - S3: Polynomial `p(r) = r + A2*r² + A3*r³ + A4*r⁴`.
   - S4: Table lookup `T[J]` and `core = T[J] * (1 + p)`.
   - S5: Exponent adjust for `2^M` and FP32 pack.

5. **Wire control and valid signals**
   - Add pipeline registers between stages.
   - Propagate `valid` bits alongside data.
   - Bypass core pipeline for special cases (S0 outputs directly).

6. **Generate a testbench against the Python model**
   - Use `exp_fp32_l64` in `exp.py` as the golden reference.
   - For each test vector `x`:
     - Compute `y_ref = exp_fp32_l64(x)` in Python.
     - Apply `x` to the RTL block, capture the output `y_rtl` once valid.
     - Compare `y_rtl` and `y_ref` bitwise or within 1 ulp.
   - Test ranges:
     - Dense samples in `[-64, 0]` (safe-softmax range).
     - Additional coverage in `[-16, 16]`.

7. **Check edge cases**
   - NaN, +Inf, −Inf.
   - Values just below and above thresholds.
   - Very small |x| (where `1 + x` path is taken).

8. **Document configuration**
   - Record the table format (FP32, addresses, ROM depth/width).
   - Record pipeline latency (cycles from input valid to output valid).
   - Record the measured max relative/absolute error vs `math.exp`.

Following this list together with the algorithm description and `exp.py` should
be enough to go from this repository to a working ASIC `exp(x)` block.
