"""
Single-precision exp(x) implementations for ASIC-oriented designs.

This module contains:
- exp_fp32:  Tang (1989) table-driven exp for IEEE-754 single precision (L=32).
- exp_fp32_l64: Improved L=64 variant with a quartic polynomial, tuned for
  use cases like safe-softmax (exp(x_i - rowmax)).

Both functions model hardware-friendly, binary calculations suitable for an ASIC
pipeline that has FP32 add/mul and a small ROM.
"""

import struct
import math

# ----------------------------------------------------------------------
# Helper functions for single-precision floating-point
# ----------------------------------------------------------------------

def float_to_hex(f):
    """Convert float to 32-bit hexadecimal string (IEEE 754 single)."""
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])[2:].upper().zfill(8)

def hex_to_float(h):
    """Convert 32-bit hexadecimal string to float."""
    # h is string like '3F800000'
    return struct.unpack('<f', struct.pack('<I', int(h, 16)))[0]

def round_to_fp32(x):
    """Round Python float (double) to nearest single-precision float."""
    # pack as single, unpack, returns float
    return struct.unpack('<f', struct.pack('<f', x))[0]

def sp(x):
    """Single-precision rounding (alias)."""
    return round_to_fp32(x)

# ----------------------------------------------------------------------
# Constants from Appendix (single precision)
# ----------------------------------------------------------------------

# Step 1 thresholds
THRESHOLD_1 = hex_to_float('435C6BBA')   # 341 * log(2) ~ 88.722...
THRESHOLD_2 = hex_to_float('33000000')   # 2^-25

# Step 2 reduction constants (Tang 1989, L = 32)
INV_L = hex_to_float('4238AA3B')         # 32 / log(2)
L1 = hex_to_float('3CB17200')            # log(2)/32 leading part
L2 = hex_to_float('333FBE8E')            # log(2)/32 trailing part

# Step 3 polynomial coefficients (Tang 1989)
A1 = hex_to_float('3F000044')
A2 = hex_to_float('3E2AAAEC')

# Step 4 table of 2^(j/32) for j = 0..31
# Each entry: (S_lead, S_trail)
S_TABLE = [
    (hex_to_float('3F800000'), hex_to_float('00000000')),  # j=0
    (hex_to_float('3F82CD80'), hex_to_float('35531585')),  # j=1
    (hex_to_float('3F85AAC0'), hex_to_float('34D9F312')),  # j=2
    (hex_to_float('3F889800'), hex_to_float('35E8092E')),  # j=3
    (hex_to_float('3F8B95C0'), hex_to_float('3471F546')),  # j=4
    (hex_to_float('3F8EA400'), hex_to_float('36E62D17')),  # j=5
    (hex_to_float('3F91C3C0'), hex_to_float('361B9D59')),  # j=6
    (hex_to_float('3F94F4C0'), hex_to_float('36BEA3FC')),  # j=7
    (hex_to_float('3F9837C0'), hex_to_float('36C14637')),  # j=8
    (hex_to_float('3F9B8D00'), hex_to_float('3666E755')),  # j=9
    (hex_to_float('3F9EF500'), hex_to_float('36C98247')),  # j=10
    (hex_to_float('3FA27040'), hex_to_float('34C0C312')),  # j=11
    (hex_to_float('3FA5FEC0'), hex_to_float('36354D8B')),  # j=12
    (hex_to_float('3FA9A140'), hex_to_float('3655A754')),  # j=13
    (hex_to_float('3FAD5800'), hex_to_float('36FBA90B')),  # j=14
    (hex_to_float('3FB123C0'), hex_to_float('36D6074B')),  # j=15
    (hex_to_float('3FB504C0'), hex_to_float('36CCCFE7')),  # j=16
    (hex_to_float('3FB8FB80'), hex_to_float('36BD1D8C')),  # j=17
    (hex_to_float('3FBD0880'), hex_to_float('368E7D60')),  # j=18
    (hex_to_float('3FC12C40'), hex_to_float('35CCA667')),  # j=19
    (hex_to_float('3FC56700'), hex_to_float('36984554')),  # j=20
    (hex_to_float('3FC9B980'), hex_to_float('36F619B9')),  # j=21
    (hex_to_float('3FCE2480'), hex_to_float('35C151F8')),  # j=22
    (hex_to_float('3FD2A800'), hex_to_float('366C8F89')),  # j=23
    (hex_to_float('3FD744C0'), hex_to_float('36F32B5A')),  # j=24
    (hex_to_float('3FDBFB80'), hex_to_float('36DE5F6C')),  # j=25
    (hex_to_float('3FE0CCC0'), hex_to_float('36776155')),  # j=26
    (hex_to_float('3FE5B900'), hex_to_float('355CEF90')),  # j=27
    (hex_to_float('3FEAC0C0'), hex_to_float('355CFBA5')),  # j=28
    (hex_to_float('3FEFE480'), hex_to_float('36E66F73')),  # j=29
    (hex_to_float('3FF52540'), hex_to_float('36F45492')),  # j=30
    (hex_to_float('3FFA8380'), hex_to_float('36CB6DC9')),  # j=31
]

# Verify table length
assert len(S_TABLE) == 32

# ----------------------------------------------------------------------
# Improved ASIC-oriented variant (conceptual model)
# L = 64, small-range polynomial
# ----------------------------------------------------------------------

L64 = 64
# Single-precision approximations of 64/log(2) and log(2)/64
INV_L_64 = sp(64.0 / math.log(2.0))
LOG2_BY_64 = sp(math.log(2.0) / 64.0)

# Simple cubic-like coefficients for exp(r) - 1 around r=0, fitted
# over |r| <= ln(2)/128 using least squares (see fitting script).
# p(r) = r + A2_64*r^2 + A3_64*r^3 + A4_64*r^4
A2_64 = sp(0.4999999999992412)          # ~0.5
A3_64 = sp(0.1666668576971548)          # ~1/6
A4_64 = sp(0.041666733294667246)        # ~1/24

# Table of 2^(j/64), j = 0..63 (single-precision model for ASIC ROM)
EXP2_TABLE_64 = [sp(math.exp(j * LOG2_BY_64)) for j in range(L64)]


def exp_fp32_l64(x: float) -> float:
    """Improved exp(x) with L=64 table and quartic polynomial (FP32 model).

    This function is a *direct software model* of an ASIC-friendly FP32 datapath:
    - 64-entry table for 2^(j/64).
    - Quartic polynomial over a very small reduced argument.
    - Input and output are IEEE-754 single precision (simulated via sp()).

    In hardware you can implement the same structure using FP32 IP blocks
    (add/mul/compare/round) and a small ROM; or, if preferred, use a wider
    fixed-point internal format but keep the same control flow and tables.
    """

    # Step 1: special cases in FP32
    if math.isnan(x):
        return float('nan')
    if math.isinf(x):
        return float('inf') if x > 0 else 0.0

    # Convert to FP32
    x = sp(x)

    # Use same thresholds as Tang for now
    if abs(x) > THRESHOLD_1:
        return float('inf') if x > 0 else 0.0
    if abs(x) < THRESHOLD_2:
        return sp(1.0 + x)

    # Step 2: range reduction for L = 64
    # N = round(x * 64/log(2)) in single precision
    tN = sp(x * INV_L_64)
    N = int(round(tN))

    # J in [0, 63], M integer
    J = ((N % L64) + L64) % L64
    M = (N - J) // L64

    # Reduced argument r ≈ x - N * log(2)/64
    # Here we use a single constant LOG2_BY_64; in hardware you would
    # typically split it into (L1 + L2) like Tang to control reduction error.
    t = sp(N * LOG2_BY_64)
    r = sp(x - t)  # r is small: |r| <= ~ln(2)/128

    # Step 3: polynomial approximation p(r) ≈ exp(r) - 1
    # p(r) = r + A2*r^2 + A3*r^3 + A4*r^4
    r2 = sp(r * r)
    r3 = sp(r2 * r)
    r4 = sp(r2 * r2)
    p = sp(r + sp(A2_64 * r2) + sp(A3_64 * r3) + sp(A4_64 * r4))

    # Step 4: reconstruction  exp(x) ≈ 2^M * 2^(J/64) * (1 + p)
    two_to_j_over_64 = EXP2_TABLE_64[J]
    one_plus_p = sp(1.0 + p)
    core = sp(two_to_j_over_64 * one_plus_p)

    # Scale by 2^M using ldexp and round back to FP32
    result = math.ldexp(core, M)
    return sp(result)

# ----------------------------------------------------------------------
# Core exp function
# ----------------------------------------------------------------------

def exp_fp32(x):
    """
    Compute exp(x) for single-precision floating-point input x.
    Returns single-precision float.
    """
    # Step 1: Filter exceptional cases
    if math.isnan(x):
        return float('nan')
    if math.isinf(x):
        if x > 0:
            return float('inf')
        else:
            return 0.0

    # Convert x to single precision (input may be double)
    x = sp(x)

    # Threshold checks
    if abs(x) > THRESHOLD_1:
        if x > 0:
            return float('inf')
        else:
            return 0.0

    if abs(x) < THRESHOLD_2:
        return sp(1.0 + x)

    # Step 2: Argument reduction
    # Compute N = round(x * INV_L) in single-precision, as in Tang 1989
    # Use single-precision multiply followed by round-to-nearest-even.
    tN = sp(x * INV_L)
    N = int(round(tN))
    # Ensure positive remainder 0-31
    N2 = ((N % 32) + 32) % 32
    N1 = N - N2

    # Compute R1 and R2 with single-precision rounding
    if abs(N) >= 2**9:
        # R1 = (X - N1*L1) - N2*L1
        t1 = sp(N1 * L1)
        t2 = sp(N2 * L1)
        R1 = sp(sp(x - t1) - t2)
    else:
        # R1 = X - N*L1
        t = sp(N * L1)
        R1 = sp(x - t)

    R2 = sp(-sp(N * L2))

    # Step 3: Polynomial approximation
    R = sp(R1 + R2)
    # Q = R*R*(A1 + R*A2)
    inner = sp(R * A2)
    inner2 = sp(A1 + inner)
    R2_sq = sp(R * R)
    Q = sp(R2_sq * inner2)
    # P = R1 + (R2 + Q)
    t = sp(R2 + Q)
    P = sp(R1 + t)

    # Step 4: Reconstruction
    J = N2
    S_lead, S_trail = S_TABLE[J]
    S = sp(S_lead + S_trail)
    t1 = sp(S * P)
    t2 = sp(S_trail + t1)
    t3 = sp(S_lead + t2)
    M = N1 // 32  # integer division
    # Multiply by 2^M using ldexp (exact)
    result = math.ldexp(t3, M)
    # Round to single precision (ldexp may produce double)
    return sp(result)

# ----------------------------------------------------------------------
# Testing and validation
# ----------------------------------------------------------------------

def test_exp():
    """Quick self-test: compare both exp variants with math.exp.

    This is a lightweight sanity check. For more detailed testing of the
    improved L=64 variant see test_exp_l64.py.
    """
    import random

    test_cases = [
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        10.0,
        -10.0,
        0.001,
        -0.001,
        88.0,  # near overflow
        -88.0,
        1e-30,
        -1e-30,
    ]

    # Add random values
    random.seed(55)
    for _ in range(1000):
        # generate values in range [-100, 100]
        x = random.uniform(-100, 100)
        test_cases.append(x)

    max_rel_32 = 0.0
    max_rel_64 = 0.0

    for x in test_cases:
        # Skip values that would overflow/underflow in single precision
        if abs(x) > 88.7:
            continue
        y_true = math.exp(x)
        y_32 = exp_fp32(x)
        y_64 = exp_fp32_l64(x)

        if y_true != 0.0:
            e32 = abs(y_32 - y_true) / abs(y_true)
            e64 = abs(y_64 - y_true) / abs(y_true)
            max_rel_32 = max(max_rel_32, e32)
            max_rel_64 = max(max_rel_64, e64)

    print(f"Tested {len(test_cases)} values")
    print(f"Max relative error (Tang L=32) : {max_rel_32:.3e}")
    print(f"Max relative error (L=64 poly): {max_rel_64:.3e}")

    # Quick sanity check for special values
    print("\nSpecial values (Tang L=32):")
    for x in [0.0, float('inf'), -float('inf'), float('nan')]:
        try:
            y = exp_fp32(x)
            print(f"exp_fp32({x}) = {y}")
        except Exception as e:
            print(f"exp_fp32({x}) raised {e}")

    print("\nSpecial values (L=64 variant):")
    for x in [0.0, float('inf'), -float('inf'), float('nan')]:
        try:
            y = exp_fp32_l64(x)
            print(f"exp_fp32_l64({x}) = {y}")
        except Exception as e:
            print(f"exp_fp32_l64({x}) raised {e}")


if __name__ == "__main__":
    test_exp()
