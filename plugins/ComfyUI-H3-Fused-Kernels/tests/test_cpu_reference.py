import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h3_fused_kernels_test", ROOT / "kernels.py")
KERNELS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(KERNELS)


def test_cpu_scale_shift_matches_native():
    torch.manual_seed(1)
    value = torch.randn(9, 7)
    shift = torch.randn(3, 7)
    scale = torch.randn(3, 7)
    segments = [(0, 2, 0), (2, 6, 2), (6, 9, 1)]
    expected = KERNELS._native_scale_shift(value.clone(), shift, scale, segments)
    actual = KERNELS.mod_scale_shift(
        value.clone(), shift, scale, segments, "auto", "accurate"
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cpu_gate_matches_native():
    torch.manual_seed(2)
    value = torch.randn(9, 7)
    other = torch.randn(9, 7)
    gate = torch.randn(3, 7)
    segments = [(0, 2, 0), (2, 6, 2), (6, 9, 1)]
    expected = KERNELS._native_gate(value.clone(), gate, other, segments)
    actual = KERNELS.mod_gate(value.clone(), gate, other, segments, "auto")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_segment_validation_rejects_gaps():
    try:
        KERNELS._segments_key([(0, 2, 0), (3, 5, 1)], 5, torch.device("cpu"))
    except ValueError:
        return
    raise AssertionError("segment validation accepted a gap")


if __name__ == "__main__":
    test_cpu_scale_shift_matches_native()
    test_cpu_gate_matches_native()
    test_segment_validation_rejects_gaps()
    print("H3 Fused Kernels CPU reference tests passed")
