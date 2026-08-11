from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch
import torch.nn.functional as F


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h3_long_sequence_nodes", ROOT / "nodes.py")
nodes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = nodes
SPEC.loader.exec_module(nodes)


class DummyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 24, bias=False)
        self.fc2 = torch.nn.Linear(12, 8, bias=False)

    def forward(self, x):
        gate, value = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * value)


class DummyAdapter:
    def __init__(self, up, down, alpha):
        self.weights = (up, down, alpha)
        self.multiplier = 0.75
        self.is_conv = False
        self._h3_long_sequence_config = {
            "profile": "16gb",
            "mlp_chunk_rows": 4096,
            "lora_chunk_mib": 32,
            "manual_reserve_gib": 0.0,
        }
        self._h3_long_sequence_original_bypass = self.original_bypass

    def original_bypass(self, org_forward, x, *args, **kwargs):
        base = org_forward(x, *args, **kwargs)
        up, down, alpha = self.weights
        scale = (alpha / down.shape[0]) * self.multiplier
        return base.add_(F.linear(F.linear(x, down), up), alpha=scale)


def test_mlp_chunking_matches_full_path():
    torch.manual_seed(1)
    mlp = DummyMLP().eval()
    x = torch.randn(50000, 8)
    expected = mlp(x)
    config = {
        "profile": "16gb_chunked",
        "mlp_chunk_rows": 4096,
        "lora_chunk_mib": 256,
        "manual_reserve_gib": 0.0,
    }
    chunked = nodes._make_chunked_mlp_forward(mlp.forward, config)
    actual = chunked(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_lora_chunking_matches_frugal_full_path():
    torch.manual_seed(2)
    x = torch.randn(50000, 8)
    base_weight = torch.randn(12, 8)
    up = torch.randn(12, 4)
    down = torch.randn(4, 8)
    adapter = DummyAdapter(up, down, 4.0)

    def base(value):
        return F.linear(value, base_weight)

    expected = adapter.original_bypass(base, x)
    original_mib = nodes._MIB
    try:
        # Keep the reference test small while forcing more than one chunk.
        nodes._MIB = 1024
        actual = nodes._chunked_lora_bypass(adapter, base, x)
    finally:
        nodes._MIB = original_mib
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_16gb_policy_only_arms_for_long_sequences():
    short = nodes._auto_policy(
        30000,
        profile="16gb",
        mlp_chunk_rows=4096,
        lora_chunk_mib=256,
        manual_reserve_gib=0.0,
    )
    long = nodes._auto_policy(
        110000,
        profile="16gb",
        mlp_chunk_rows=4096,
        lora_chunk_mib=256,
        manual_reserve_gib=0.0,
    )
    assert short.reserve_bytes == 0
    assert long.reserve_bytes == 5 * nodes._GIB
    assert short.threshold_rows == long.threshold_rows == 49152
    assert not long.mlp_chunking

    fallback = nodes._auto_policy(
        110000,
        profile="16gb_chunked",
        mlp_chunk_rows=4096,
        lora_chunk_mib=256,
        manual_reserve_gib=0.0,
    )
    assert fallback.mlp_chunking


def test_plugin_registration():
    package_spec = importlib.util.spec_from_file_location(
        "h3_long_sequence_plugin",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    package = importlib.util.module_from_spec(package_spec)
    assert package_spec.loader is not None
    sys.modules[package_spec.name] = package
    package_spec.loader.exec_module(package)
    assert "H3LongSequenceVRAMOptimizer" in package.NODE_CLASS_MAPPINGS
    node = package.NODE_CLASS_MAPPINGS["H3LongSequenceVRAMOptimizer"]
    profiles = node.INPUT_TYPES()["required"]["profile"][0]
    assert profiles == ["auto", "16gb", "16gb_chunked", "24gb_plus", "off"]


if __name__ == "__main__":
    test_mlp_chunking_matches_full_path()
    test_lora_chunking_matches_frugal_full_path()
    test_16gb_policy_only_arms_for_long_sequences()
    test_plugin_registration()
    print("H3 Long-Sequence reference tests passed")
