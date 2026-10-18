"""Deterministic integer L2 clipping in plaintext or CryptGPU 3PC.

For Q16 input x, C_int=floor(C*65536), r=max(C_int, ceil(sqrt(sum(x*x)))),
a=floor(2**30*C_int/r), and y=sign(x)*floor(abs(x)*a/2**30).
Then ||y||_2 <= a*||x||_2/2**30 <= C_int, including rounding.
MPC arithmetic uses precision=0 with exact comparisons and public-power-of-two
division. The caller enforces sum(x*x)<2**56.
"""

import math
import time

import numpy as np

FACTOR_BITS = 30
FACTOR_SCALE = 1 << FACTOR_BITS
NORM_BITS = 28
LOCAL_INTEGER_CLIP = "local_l2_integer_then_mpc_sum"
MPC_INTEGER_CLIP = "mpc_l2_integer"
MPC_NEAREST_CLIP = "mpc_l2_integer_nearest"
MPC_CLIP_MODES = (MPC_INTEGER_CLIP, MPC_NEAREST_CLIP)
INTEGER_CLIP_MODES = (LOCAL_INTEGER_CLIP,) + MPC_CLIP_MODES


def nearest_inner_radius(radius, dimension):
    """Reserve ceil(sqrt(d)/2) integer units for nearest rounding's L2 error."""
    if not 1 <= radius < (1 << NORM_BITS) or dimension < 1:
        raise ValueError("invalid clipping radius or dimension")
    margin = math.isqrt(dimension) // 2
    margin += int(4 * margin * margin < dimension)
    if radius <= margin:
        raise ValueError("clipping radius is too small for certified nearest rounding")
    return radius - margin


def integer_l2_nearest_reference(values, radius):
    """Round to nearest, with ties toward +infinity and a norm-bounded inner radius."""
    x = np.asarray(values, dtype=np.int64)
    inner = nearest_inner_radius(radius, x.size)
    max_abs = max(abs(int(x.min(initial=0))), abs(int(x.max(initial=0))))
    if x.size * max_abs * max_abs >= 1 << (2 * NORM_BITS):
        raise OverflowError("integer clipping input exceeds the public ring domain")
    norm_sq = int(np.sum(x * x, dtype=np.int64))
    root = math.isqrt(norm_sq)
    denominator = max(inner, root + int(root * root < norm_sq))
    factor = FACTOR_SCALE * inner // denominator
    return (x * factor + FACTOR_SCALE // 2) // FACTOR_SCALE


def integer_l2_clip_reference(values, radius):
    """Compute the exact integer reference for clipping."""
    x = np.asarray(values, dtype=np.int64)
    # The accepted input domain guarantees these int64 products/sums fit.
    if not 1 <= radius < (1 << NORM_BITS):
        raise ValueError("integer clipping radius must be in [1, 2**28)")
    max_abs = max(abs(int(x.min(initial=0))), abs(int(x.max(initial=0))))
    if x.size * max_abs * max_abs >= 1 << (2 * NORM_BITS):
        raise OverflowError("integer clipping input exceeds the public ring domain")
    norm_sq = int(np.sum(x * x, dtype=np.int64))
    root = math.isqrt(norm_sq)
    denominator = max(radius, root + int(root * root < norm_sq))
    factor = FACTOR_SCALE * radius // denominator
    return np.sign(x) * ((np.abs(x) * factor) // FACTOR_SCALE)


def _raw(tensor):
    """Discard implicit encoders returned by upstream OT primitives."""
    from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor

    return ArithmeticSharedTensor.from_shares(tensor.share, precision=0)


def _bit_to_arithmetic(bit):
    import torch
    from crypten.mpc.primitives import resharing

    ones = torch.ones_like(bit.share.data) if bit.rank == 0 else None
    return _raw(resharing.mixed_mul_scalar(ones, bit, bits=1))


def _negative(value):
    from crypten.mpc.primitives.converters import get_msb

    return _bit_to_arithmetic(get_msb(value))


def _floor_nonnegative_power_two(value, bits, one_sided=False):
    """Compute floor(value/2**bits) on secret shares.

    The general path requires 0<=value<2**58 and quotient<2**28.
    The one-sided path requires bits=30, 0<=value<2**60 and quotient<2**30.

    Rank 0 holds a=(s0+s2) mod 2**64 and rank 1 holds b=s1. Their signed high
    and unsigned low parts are shared separately. A secret carry correction
    and modular centering recover the exact quotient within these bounds.
    """
    import torch
    from crypten.mpc.primitives import resharing
    from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor

    rep = resharing.replicate_shares(value.share)
    if value.rank == 0:
        summand = value.share + rep
    elif value.rank == 1:
        summand = value.share
    else:
        summand = torch.zeros_like(value.share)
    low = ArithmeticSharedTensor.from_shares(summand & ((1 << bits) - 1), precision=0)
    high = ArithmeticSharedTensor.from_shares(summand >> bits, precision=0)
    # Re-randomize high and low sharings with independent PRZS masks before
    # replication; otherwise the receiving rank can recover the private summands.
    low = low + _raw(ArithmeticSharedTensor.PRZS(low.size(), device=value.device))
    high = high + _raw(ArithmeticSharedTensor.PRZS(high.size(), device=value.device))
    quotient = high + (1 - _negative(low - (1 << bits)))
    modulus = 1 << (64 - bits)
    if one_sided:
        # For 0<=v<2**60 and bits=30, a+b is v or v-2**64.
        # The quotient is q or q-modulus with q<2**30, so one sign correction suffices.
        if bits != FACTOR_BITS:
            raise ValueError("one-sided truncation requires 30 fractional bits")
        return quotient + _negative(quotient) * modulus
    negative_wrap = _negative(quotient + modulus // 2)
    positive_wrap = 1 - _negative(quotient - modulus // 2)
    return quotient + negative_wrap * modulus - positive_wrap * modulus


def _l2_factor(norm_sq, radius):
    # Bitwise integer square root, with a public, input-independent loop count.
    root = norm_sq * 0
    for bit in reversed(range(NORM_BITS)):
        candidate = root + (1 << bit)
        fits = 1 - _negative(norm_sq - candidate * candidate)
        root = root + fits * (1 << bit)
    ceil_root = root + _negative(root * root - norm_sq)
    denominator = ceil_root + _negative(ceil_root - radius) * (radius - ceil_root)

    # Long division of a public numerator by the secret denominator.
    factor = root * 0
    numerator = FACTOR_SCALE * radius
    for bit in reversed(range(FACTOR_BITS + 1)):
        candidate = factor + (1 << bit)
        fits = 1 - _negative(denominator * candidate * -1 + numerator)
        factor = factor + fits * (1 << bit)
    return factor


def cryptgpu_l2_clip(deltas, radius, chunk_size=32768, mode=MPC_INTEGER_CLIP):
    """Clip K x d arithmetic shares, returning shares and public timings only."""
    import torch
    from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor

    if not 1 <= radius < (1 << NORM_BITS) or chunk_size < 1:
        raise ValueError("invalid integer L2 clipping parameters")
    if int(deltas.encoder.scale) != 1:
        raise ValueError("integer L2 clipping requires precision=0")
    if mode not in MPC_CLIP_MODES:
        raise ValueError("unknown secure integer clipping mode")
    inner = nearest_inner_radius(radius, deltas.size(1)) if mode == MPC_NEAREST_CLIP else radius

    def stamp():
        if deltas.device.type == "cuda":
            torch.cuda.synchronize(deltas.device)
        return time.perf_counter()

    start = stamp()
    norm_sq = deltas[:, :1] * 0
    for offset in range(0, deltas.size(1), chunk_size):
        block = ArithmeticSharedTensor.from_shares(
            deltas.share[:, offset : offset + chunk_size].contiguous(), precision=0
        )
        norm_sq = norm_sq + (block * block).sum(dim=1, keepdim=True)
    norm_done = stamp()

    factor = _l2_factor(norm_sq, inner)
    factor_done = stamp()

    output_share = torch.empty_like(deltas.share)
    for offset in range(0, deltas.size(1), chunk_size):
        block = ArithmeticSharedTensor.from_shares(
            deltas.share[:, offset : offset + chunk_size].contiguous(), precision=0
        )
        if mode == MPC_NEAREST_CLIP:
            # |x*a| < 2**58. A public multiple of Q makes the dividend
            # nonnegative (<2**60), avoiding secret abs/sign restoration.
            offset_int = 1 << 58
            shifted = block * factor + (offset_int + FACTOR_SCALE // 2)
            rounded = _floor_nonnegative_power_two(shifted, FACTOR_BITS, one_sided=True)
            output_share[:, offset : offset + chunk_size] = (
                rounded - offset_int // FACTOR_SCALE
            ).share
        else:
            sign = 1 - _negative(block) * 2
            magnitude = block * sign
            product = magnitude * factor
            clipped_magnitude = _floor_nonnegative_power_two(product, FACTOR_BITS)
            output_share[:, offset : offset + chunk_size] = (clipped_magnitude * sign).share
    end = stamp()
    return ArithmeticSharedTensor.from_shares(output_share, precision=0), {
        "mpc_clip_norm_s": norm_done - start,
        "mpc_clip_factor_s": factor_done - norm_done,
        "mpc_clip_rescale_s": end - factor_done,
        "mpc_clip_s": end - start,
    }
