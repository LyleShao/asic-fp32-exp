#!/usr/bin/env python3
"""
gen_vectors.py
Generate golden reference test vectors for the FP32 IP cores.
Uses Python's struct.pack('<f', ...) for bit-exact IEEE-754 encoding.

Outputs (in asic/ip/tb/):
  vec_int_to_fp32.hex    – 16-bit int input, 32-bit FP32 output
  vec_fp32_to_int.hex    – 32-bit FP32 input, 16-bit int output
  vec_fp32_add.hex       – 32+32-bit inputs, 32-bit output
  vec_fp32_mul.hex       – 32+32-bit inputs, 32-bit output
  vec_fp32_sub.hex       – 32+32-bit inputs, 32-bit output

Format: one test vector per line, hex digits, no prefix.
  int_to_fp32: "IIII YYYYYYYY\n"  (4-digit int, 8-digit fp32)
  fp32_to_int: "AAAAAAAA IIII\n"  (8-digit fp32, 4-digit int signed)
  fp32_add/sub/mul: "AAAAAAAA BBBBBBBB YYYYYYYY\n"
"""

import struct
import math
import os

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def f2b(v: float) -> int:
    """Pack float v as IEEE-754 single and return its bit pattern (uint32)."""
    return struct.unpack('<I', struct.pack('<f', v))[0]


def b2f(bits: int) -> float:
    """Interpret bits (uint32) as IEEE-754 single and return float."""
    return struct.unpack('<f', struct.pack('<I', bits & 0xFFFF_FFFF))[0]


def is_nan(bits: int) -> bool:
    exp  = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    return (exp == 0xFF) and (mant != 0)


def is_inf(bits: int) -> bool:
    exp  = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    return (exp == 0xFF) and (mant == 0)


def is_denormal(bits: int) -> bool:
    exp  = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    return (exp == 0) and (mant != 0)


def safe_pack_f32(v: float) -> int:
    """Pack a Python float as FP32, handling overflow to ±Inf."""
    if math.isnan(v):
        return 0x7FC00000  # canonical qNaN
    if math.isinf(v) or abs(v) > 3.4028235e+38:
        # Overflow → ±Inf
        return 0xFF800000 if v < 0 else 0x7F800000
    return struct.unpack('<I', struct.pack('<f', v))[0]


def fp32_add_golden(a_bits: int, b_bits: int) -> int:
    """
    Compute a + b in FP32 with:
      - FTZ: denormal inputs → 0 before add
      - Python float arithmetic (double-precision intermediary, but we
        immediately round back to float32 so result is correctly rounded)
    """
    # FTZ input
    if is_denormal(a_bits):
        a_bits = a_bits & 0x8000_0000  # ±0
    if is_denormal(b_bits):
        b_bits = b_bits & 0x8000_0000  # ±0

    a = b2f(a_bits)
    b = b2f(b_bits)

    result = safe_pack_f32(a + b)

    # FTZ output: flush denormal result to ±0
    if is_denormal(result):
        result = result & 0x8000_0000
    return result


def fp32_mul_golden(a_bits: int, b_bits: int) -> int:
    if is_denormal(a_bits):
        a_bits = a_bits & 0x8000_0000
    if is_denormal(b_bits):
        b_bits = b_bits & 0x8000_0000

    a = b2f(a_bits)
    b = b2f(b_bits)

    result = safe_pack_f32(a * b)

    if is_denormal(result):
        result = result & 0x8000_0000
    return result


def fp32_sub_golden(a_bits: int, b_bits: int) -> int:
    if is_denormal(a_bits):
        a_bits = a_bits & 0x8000_0000
    if is_denormal(b_bits):
        b_bits = b_bits & 0x8000_0000

    a = b2f(a_bits)
    b = b2f(b_bits)

    result = safe_pack_f32(a - b)

    if is_denormal(result):
        result = result & 0x8000_0000
    return result


def int_to_fp32_golden(i: int) -> int:
    """Signed 16-bit integer → FP32 bit pattern. Exact (no rounding needed)."""
    assert -32768 <= i <= 32767
    bits = struct.unpack('<I', struct.pack('<f', float(i)))[0]
    return bits


def fp32_to_int_golden(a_bits: int) -> int:
    """
    FP32 → signed 16-bit integer, RNE, saturate, FTZ denormals → 0.
    Returns integer in range [-32768, 32767].
    """
    if is_nan(a_bits) or is_inf(a_bits) or ((a_bits & 0x7FFFFFFF) == 0):
        return 0
    if is_denormal(a_bits):
        return 0

    sign = (a_bits >> 31) & 1
    exp  = (a_bits >> 23) & 0xFF
    mant = a_bits & 0x7FFFFF
    true_exp = exp - 127

    if true_exp < 0:
        # |value| < 1.0; can still round to ±1 via RNE
        # G bit is sig[22] when right-shifted by 23 positions (true_exp = -1 → 1 shift)
        # For simplicity use Python rounding:
        v = b2f(a_bits)
        # Python's round() uses banker's rounding (RNE)
        i = round(v)
        return max(-32768, min(32767, int(i)))

    # Use Python round() which is RNE
    v = b2f(a_bits)
    i = round(v)
    return max(-32768, min(32767, int(i)))


# ---------------------------------------------------------------------------
# Vector generation
# ---------------------------------------------------------------------------

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def write_int_to_fp32():
    path = os.path.join(OUT_DIR, 'vec_int_to_fp32.hex')
    with open(path, 'w') as f:
        for i in range(-32768, 32768):
            i16 = i & 0xFFFF
            out = int_to_fp32_golden(i)
            f.write(f'{i16:04x} {out:08x}\n')
    print(f'Wrote {path} ({65536} vectors)')


def write_fp32_to_int():
    """
    Use a representative set of FP32 inputs:
      - Exact integers -32768..32767
      - Some near-boundary values, half-integers (RNE tests)
      - ±0, ±Inf, NaN, denormals
      - Large values (overflow saturation)
    """
    vectors = []
    # Exact integers in range
    for i in range(-32768, 32768):
        bits = f2b(float(i))
        vectors.append((bits, fp32_to_int_golden(bits)))

    # Half-integers (RNE): n.5 for n in -100..100
    for n in range(-100, 101):
        v = n + 0.5
        bits = f2b(v)
        vectors.append((bits, fp32_to_int_golden(bits)))

    # Special values
    specials = [
        0x00000000,  # +0
        0x80000000,  # -0
        0x7F800000,  # +Inf
        0xFF800000,  # -Inf
        0x7FC00000,  # qNaN
        0x00000001,  # smallest denormal
        0x007FFFFF,  # largest denormal
        0x47000000,  # 32768.0
        0xC7000000,  # -32768.0
        0x46FFFE00,  # 32767.0
        0x47000100,  # 32768.5
        0xC7000100,  # -32768.5
    ]
    for bits in specials:
        vectors.append((bits, fp32_to_int_golden(bits)))

    path = os.path.join(OUT_DIR, 'vec_fp32_to_int.hex')
    with open(path, 'w') as f:
        for (a_bits, result) in vectors:
            r16 = result & 0xFFFF
            f.write(f'{a_bits:08x} {r16:04x}\n')
    print(f'Wrote {path} ({len(vectors)} vectors)')


def write_fp32_add_sub_mul():
    pairs = []

    # Basic arithmetic pairs (exact representable values)
    test_vals = [
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0,
        1.5, -1.5, 3.0, -3.0, 100.0, -100.0,
        1.0/3.0, math.pi, math.e,
    ]
    for a in test_vals:
        for b in test_vals:
            pairs.append((f2b(a), f2b(b)))

    # Special cases
    specials_1d = [
        0x7F800000,  # +Inf
        0xFF800000,  # -Inf
        0x7FC00000,  # qNaN
        0x00000000,  # +0
        0x80000000,  # -0
        0x00000001,  # denormal
        0x7F7FFFFF,  # MAX_FLOAT
    ]
    for s in specials_1d:
        for v in [f2b(1.0), f2b(-1.0), f2b(0.0), s]:
            pairs.append((s, v))
            pairs.append((v, s))

    # Cancellation: a + (-a)
    for v in [1.0, -1.0, 100.0, 1.5]:
        pairs.append((f2b(v), f2b(-v)))

    # Overflow
    mx = 0x7F7FFFFF
    pairs.append((mx, mx))

    # RNE half-way case: 1.0 + 2^-24 (tie to even)
    pairs.append((f2b(1.0), f2b(2**-24)))
    # 1.0 + 2^-23 (rounds up)
    pairs.append((f2b(1.0), f2b(2**-23)))

    # Deduplicate
    pairs = list(dict.fromkeys(pairs))

    add_path = os.path.join(OUT_DIR, 'vec_fp32_add.hex')
    sub_path = os.path.join(OUT_DIR, 'vec_fp32_sub.hex')
    mul_path = os.path.join(OUT_DIR, 'vec_fp32_mul.hex')

    with open(add_path, 'w') as fa, \
         open(sub_path, 'w') as fs, \
         open(mul_path, 'w') as fm:
        for (a_bits, b_bits) in pairs:
            add_out = fp32_add_golden(a_bits, b_bits)
            sub_out = fp32_sub_golden(a_bits, b_bits)
            mul_out = fp32_mul_golden(a_bits, b_bits)
            fa.write(f'{a_bits:08x} {b_bits:08x} {add_out:08x}\n')
            fs.write(f'{a_bits:08x} {b_bits:08x} {sub_out:08x}\n')
            fm.write(f'{a_bits:08x} {b_bits:08x} {mul_out:08x}\n')

    n = len(pairs)
    print(f'Wrote {add_path} ({n} vectors)')
    print(f'Wrote {sub_path} ({n} vectors)')
    print(f'Wrote {mul_path} ({n} vectors)')


if __name__ == '__main__':
    write_int_to_fp32()
    write_fp32_to_int()
    write_fp32_add_sub_mul()
    print('Done.')
