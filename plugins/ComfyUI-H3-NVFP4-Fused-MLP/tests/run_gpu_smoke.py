"""Numerical and preprocessing microbenchmark for the fused H3 NVFP4 path."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import comfy_kitchen as ck  # noqa: E402
from kernels import (  # noqa: E402
    fused_swiglu_quantize_nvfp4,
    materialize_swiglu_exact,
)


def native(raw):
    gate, up = raw.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate).mul_(up)
    scale = (torch.amax(activated.abs()) / (448.0 * 6.0)).to(torch.float32)
    packed, block_scales = ck.quantize_nvfp4(
        activated, scale, pad_16x=(raw.shape[0] % 16 != 0)
    )
    return packed, scale.reshape(1), block_scales


def one_kernel_activation(raw):
    activated = materialize_swiglu_exact(raw)
    scale = (torch.amax(activated.abs()) / (448.0 * 6.0)).to(torch.float32)
    packed, block_scales = ck.quantize_nvfp4(
        activated, scale, pad_16x=(raw.shape[0] % 16 != 0)
    )
    return packed, scale.reshape(1), block_scales


def milliseconds(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--width", type=int, default=14336)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(1234)
    raw = torch.randn(
        (args.rows, args.width * 2), device="cuda", dtype=torch.bfloat16
    )
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    print("shape", tuple(raw.shape), "MiB", raw.nbytes / 1048576)

    ref_q, ref_scale, ref_bs = native(raw)
    materialized_q, materialized_scale, materialized_bs = one_kernel_activation(raw)
    torch.cuda.synchronize()
    print(
        "one_kernel_activation_exact",
        "q_equal=", torch.equal(materialized_q, ref_q),
        "scale_equal=", torch.equal(materialized_scale, ref_scale),
        "scales_equal=", torch.equal(
            materialized_bs.view(torch.uint8), ref_bs.view(torch.uint8)
        ),
    )
    for precision in ("native_rounding", "fast_fp32"):
        q, scale, bs, shape = fused_swiglu_quantize_nvfp4(
            raw, precision=precision
        )
        torch.cuda.synchronize()
        q_equal = torch.equal(q, ref_q)
        bs_equal = torch.equal(bs.view(torch.uint8), ref_bs.view(torch.uint8))
        q_match = (q == ref_q).float().mean().item()
        bs_match = (
            bs.view(torch.uint8) == ref_bs.view(torch.uint8)
        ).float().mean().item()
        print(
            precision,
            "shape=", shape,
            "scale=", scale.item(),
            "ref_scale=", ref_scale.item(),
            "q_equal=", q_equal,
            "q_match=", q_match,
            "scales_equal=", bs_equal,
            "scale_match=", bs_match,
        )

    cached_q, cached_scale, cached_bs, _ = fused_swiglu_quantize_nvfp4(
        raw, precision="native_rounding", tensor_scale=ref_scale
    )
    torch.cuda.synchronize()
    print(
        "cached_exact_scale",
        "q_equal=", torch.equal(cached_q, ref_q),
        "scales_equal=", torch.equal(
            cached_bs.view(torch.uint8), ref_bs.view(torch.uint8)
        ),
    )

    native_ms = milliseconds(
        lambda: native(raw), args.warmup, args.iterations
    )
    fused_ms = milliseconds(
        lambda: fused_swiglu_quantize_nvfp4(raw, precision="native_rounding"),
        args.warmup,
        args.iterations,
    )
    cached_ms = milliseconds(
        lambda: fused_swiglu_quantize_nvfp4(
            raw, precision="native_rounding", tensor_scale=ref_scale
        ),
        args.warmup,
        args.iterations,
    )
    materialized_ms = milliseconds(
        lambda: one_kernel_activation(raw), args.warmup, args.iterations
    )
    print(
        f"native={native_ms:.3f} ms fused={fused_ms:.3f} ms "
        f"speedup={native_ms / fused_ms:.3f}x "
        f"cached={cached_ms:.3f} ms cached_speedup={native_ms / cached_ms:.3f}x "
        f"one_kernel_activation={materialized_ms:.3f} ms "
        f"one_kernel_speedup={native_ms / materialized_ms:.3f}x"
    )


if __name__ == "__main__":
    main()
