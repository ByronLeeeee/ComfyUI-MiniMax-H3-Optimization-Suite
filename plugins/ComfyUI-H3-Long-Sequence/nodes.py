from __future__ import annotations

import logging
import math
import types
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.patcher_extension
from comfy.ldm.minimax.model import MiniMaxH3Model


LOG = logging.getLogger("h3_long_sequence")
CONFIG_KEY = "h3_long_sequence_vram_config"
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_ANNOUNCED_MLP: set[tuple[int, int]] = set()
_ANNOUNCED_LORA: set[tuple[int, int, int]] = set()
_ANNOUNCED_RESERVE: set[tuple[int, int]] = set()
_ANNOUNCED_WARNINGS: set[str] = set()


@dataclass(frozen=True)
class RuntimePolicy:
    threshold_rows: int
    mlp_chunk_rows: int
    mlp_chunking: bool
    lora_chunk_mib: int
    reserve_bytes: int


def _h3_model(model):
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, MiniMaxH3Model):
        raise ValueError(
            "H3 Long-Sequence VRAM Optimizer only supports ComfyUI's native "
            "MiniMax H3 model."
        )
    return diffusion_model


def _total_vram_gib() -> float:
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / _GIB
    except Exception:
        pass
    # Auto is intentionally conservative when hardware discovery is unavailable.
    return 16.0


def _target_rows_from_noise(noise_shape) -> int:
    """Conservative target token estimate from H3's packed [B, 1, N] latent.

    A target video token contains 24 channels x a 2x2 patch = 96 packed
    scalars. Audio tokens contain fewer values, so division by 96 under-counts
    rather than over-counts the amount of long-sequence work.
    """

    if noise_shape is None or len(noise_shape) != 3 or int(noise_shape[1]) != 1:
        return 0
    return math.prod(int(v) for v in noise_shape) // 96


def _auto_policy(
    rows: int,
    *,
    profile: str,
    mlp_chunk_rows: int,
    lora_chunk_mib: int,
    manual_reserve_gib: float,
) -> RuntimePolicy:
    vram_gib = _total_vram_gib()
    if profile == "16gb":
        vram_gib = 16.0
    elif profile == "16gb_chunked":
        vram_gib = 16.0
    elif profile == "24gb_plus":
        vram_gib = 24.0

    if vram_gib <= 18.0:
        threshold = 49152
        # At roughly 110k target rows the native QKV and FC1 transients are
        # several GiB each. Keep progressively more activation headroom.
        if rows < threshold:
            reserve = 0
        elif rows < 80000:
            reserve = 3 * _GIB
        else:
            reserve = 5 * _GIB
    elif vram_gib <= 26.0:
        threshold = 73728
        reserve = 0 if rows < threshold else (2 * _GIB if rows < 110000 else 3 * _GIB)
    else:
        threshold = 98304
        reserve = 0 if rows < threshold else 2 * _GIB

    if manual_reserve_gib > 0:
        reserve = int(manual_reserve_gib * _GIB)

    return RuntimePolicy(
        threshold_rows=threshold,
        mlp_chunk_rows=max(256, int(mlp_chunk_rows)),
        # NVFP4 dynamically derives an input scale from all rows. Chunking an
        # MLP therefore uses per-chunk scales and is not bit/numerically exact.
        # Keep it as an explicit OOM fallback instead of silently enabling it.
        mlp_chunking=profile == "16gb_chunked",
        lora_chunk_mib=max(32, int(lora_chunk_mib)),
        reserve_bytes=int(reserve),
    )


def _make_chunked_mlp_forward(previous_forward, config: dict):
    def forward(self, x):
        # H3 inference is a packed two-dimensional sequence. Avoid changing
        # training/autograd or an unknown future layout.
        if x.ndim != 2 or x.requires_grad:
            return previous_forward(x)

        rows = int(x.shape[0])
        policy = _auto_policy(
            rows,
            profile=config["profile"],
            mlp_chunk_rows=config["mlp_chunk_rows"],
            lora_chunk_mib=config["lora_chunk_mib"],
            manual_reserve_gib=config["manual_reserve_gib"],
        )
        if (
            not policy.mlp_chunking
            or rows < policy.threshold_rows
            or rows <= policy.mlp_chunk_rows
        ):
            return previous_forward(x)

        out_features = int(getattr(self.fc2, "out_features", x.shape[-1]))
        output = torch.empty(
            (rows, out_features), dtype=x.dtype, device=x.device
        )
        for start in range(0, rows, policy.mlp_chunk_rows):
            stop = min(rows, start + policy.mlp_chunk_rows)
            chunk = previous_forward(x[start:stop])
            output[start:stop].copy_(chunk)
            del chunk

        key = (rows, policy.mlp_chunk_rows)
        if key not in _ANNOUNCED_MLP:
            _ANNOUNCED_MLP.add(key)
            full_mib = rows * 28672 * x.element_size() / _MIB
            chunk_mib = min(rows, policy.mlp_chunk_rows) * 28672 * x.element_size() / _MIB
            LOG.info(
                "H3 long-sequence MLP chunking active: rows=%d, chunk_rows=%d; "
                "FC1 BF16/FP16 output upper bound %.1f -> %.1f MiB per call.",
                rows,
                policy.mlp_chunk_rows,
                full_mib,
                chunk_mib,
            )
        return output

    return types.MethodType(forward, previous_forward.__self__)


def _chunked_lora_bypass(self, org_forward, x, *args, **kwargs):
    config = getattr(self, "_h3_long_sequence_config", None)
    original = getattr(self, "_h3_long_sequence_original_bypass", None)
    if (
        config is None
        or original is None
        or getattr(self, "is_conv", False)
        or x.ndim != 2
        or x.requires_grad
    ):
        return original(org_forward, x, *args, **kwargs)

    rows = int(x.shape[0])
    policy = _auto_policy(
        rows,
        profile=config["profile"],
        mlp_chunk_rows=config["mlp_chunk_rows"],
        lora_chunk_mib=config["lora_chunk_mib"],
        manual_reserve_gib=config["manual_reserve_gib"],
    )
    if rows < policy.threshold_rows:
        return original(org_forward, x, *args, **kwargs)

    up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
    output_width = int(up.shape[0])
    element_size = x.element_size()
    target_bytes = policy.lora_chunk_mib * _MIB
    chunk_rows = max(128, target_bytes // max(1, output_width * element_size))
    chunk_rows = max(128, (chunk_rows // 128) * 128)
    if chunk_rows >= rows:
        return original(org_forward, x, *args, **kwargs)

    base_out = org_forward(x, *args, **kwargs)
    rank = int(down.shape[0])
    scale = (alpha / rank if alpha is not None else 1.0) * getattr(
        self, "multiplier", 1.0
    )
    down = down.to(dtype=x.dtype)
    up = up.to(dtype=x.dtype)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        delta = F.linear(F.linear(x[start:stop], down), up)
        base_out[start:stop].add_(delta, alpha=scale)
        del delta

    key = (rows, output_width, chunk_rows)
    if key not in _ANNOUNCED_LORA:
        _ANNOUNCED_LORA.add(key)
        full_mib = rows * output_width * element_size / _MIB
        chunk_mib = chunk_rows * output_width * element_size / _MIB
        LOG.info(
            "H3 Turbo LoRA chunking active: rows=%d, out=%d, chunk_rows=%d; "
            "delta temporary %.1f -> %.1f MiB.",
            rows,
            output_width,
            chunk_rows,
            full_mib,
            chunk_mib,
        )
    return base_out


def _install_turbo_lora_chunking(diffusion_model, config: dict) -> tuple[int, int]:
    active = 0
    installed = 0
    for module in diffusion_model.modules():
        owner = getattr(getattr(module, "forward", None), "__self__", None)
        adapter = getattr(owner, "adapter", None)
        if adapter is None or type(adapter).__name__ != "_FrugalLoRA":
            continue
        active += 1
        if not hasattr(adapter, "_h3_long_sequence_original_bypass"):
            adapter._h3_long_sequence_original_bypass = adapter.bypass_forward
            adapter.bypass_forward = types.MethodType(_chunked_lora_bypass, adapter)
            installed += 1
        adapter._h3_long_sequence_config = config
    return active, installed


def _lora_install_wrapper(diffusion_model, config: dict):
    state = {"installed": False}

    def wrapper(executor, *args, **kwargs):
        if not state["installed"]:
            active, installed = _install_turbo_lora_chunking(diffusion_model, config)
            if active:
                state["installed"] = True
                LOG.info(
                    "H3 Long-Sequence armed chunked bypass on %d Turbo LoRA "
                    "adapters (%d newly installed).",
                    active,
                    installed,
                )
        return executor(*args, **kwargs)

    return wrapper


def _inflated_noise_shape(model, noise_shape, conds, reserve_bytes: int):
    import comfy.sampler_helpers

    _, base_minimum = comfy.sampler_helpers.estimate_memory(model, noise_shape, conds)
    target = int(base_minimum + reserve_bytes)
    original_width = int(noise_shape[-1])
    adjusted = list(noise_shape)

    low = original_width + 1
    high = original_width
    high_minimum = base_minimum
    for _ in range(8):
        high *= 2
        adjusted[-1] = high
        _, high_minimum = comfy.sampler_helpers.estimate_memory(model, adjusted, conds)
        if high_minimum >= target:
            break
    if high_minimum <= base_minimum:
        return None

    best_width = high
    best_minimum = high_minimum
    while low <= high:
        candidate = (low + high) // 2
        adjusted[-1] = candidate
        _, candidate_minimum = comfy.sampler_helpers.estimate_memory(
            model, adjusted, conds
        )
        if candidate_minimum >= target:
            best_width = candidate
            best_minimum = candidate_minimum
            high = candidate - 1
        else:
            low = candidate + 1
    adjusted[-1] = best_width
    return adjusted, int(base_minimum), int(best_minimum)


def _activation_reserve_wrapper(config: dict):
    def wrapper(executor, model, noise_shape, conds, *args, **kwargs):
        rows = _target_rows_from_noise(noise_shape)
        policy = _auto_policy(
            rows,
            profile=config["profile"],
            mlp_chunk_rows=config["mlp_chunk_rows"],
            lora_chunk_mib=config["lora_chunk_mib"],
            manual_reserve_gib=config["manual_reserve_gib"],
        )
        if rows < policy.threshold_rows or policy.reserve_bytes <= 0:
            return executor(model, noise_shape, conds, *args, **kwargs)

        try:
            inflated = _inflated_noise_shape(
                model, noise_shape, conds, policy.reserve_bytes
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if reason not in _ANNOUNCED_WARNINGS:
                _ANNOUNCED_WARNINGS.add(reason)
                LOG.warning("H3 activation reserve disabled for this run: %s", reason)
            return executor(model, noise_shape, conds, *args, **kwargs)
        if inflated is None:
            return executor(model, noise_shape, conds, *args, **kwargs)

        adjusted, before_minimum, after_minimum = inflated
        resident_before = int(model.loaded_size())
        result = executor(model, adjusted, conds, *args, **kwargs)
        resident_after = int(model.loaded_size())
        actual = max(0, after_minimum - before_minimum)
        key = (rows, round(actual / _MIB))
        if key not in _ANNOUNCED_RESERVE:
            _ANNOUNCED_RESERVE.add(key)
            LOG.info(
                "H3 long-sequence activation reserve: rows~%d, requested %.1f MiB, "
                "estimator %.1f -> %.1f MiB; loader-tracked resident weights "
                "%.1f -> %.1f MiB. The real latent is unchanged.",
                rows,
                policy.reserve_bytes / _MIB,
                before_minimum / _MIB,
                after_minimum / _MIB,
                resident_before / _MIB,
                resident_after / _MIB,
            )
        return result

    return wrapper


class H3LongSequenceVRAMOptimizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "profile": (
                    ["auto", "16gb", "16gb_chunked", "24gb_plus", "off"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "mlp_chunk_rows": (
                    "INT",
                    {"default": 4096, "min": 256, "max": 32768, "step": 256},
                ),
                "lora_chunk_mib": (
                    "INT",
                    {"default": 256, "min": 32, "max": 1024, "step": 32},
                ),
                "manual_reserve_gib": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 12.0, "step": 0.5},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/minimax_h3/suite"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Reduces MiniMax H3 long-sequence peak VRAM without reducing steps or "
        "splitting the video timeline. Exact profiles chunk the runtime Turbo-LoRA "
        "delta and leave activation headroom in ComfyUI's dynamic weight loader; "
        "the explicit 16gb_chunked fallback also chunks token-wise MLP work. Auto "
        "bypasses short sequences."
    )

    def apply(
        self,
        model,
        profile="auto",
        mlp_chunk_rows=4096,
        lora_chunk_mib=256,
        manual_reserve_gib=0.0,
    ):
        if profile == "off":
            return (model,)
        if profile not in ("auto", "16gb", "16gb_chunked", "24gb_plus"):
            raise ValueError(f"unknown H3 long-sequence profile: {profile}")

        patched = model.clone()
        diffusion_model = _h3_model(patched)
        transformer_options = patched.model_options.setdefault(
            "transformer_options", {}
        )
        if CONFIG_KEY in transformer_options:
            raise ValueError("H3 Long-Sequence VRAM Optimizer has already been applied.")

        config = {
            "profile": profile,
            "mlp_chunk_rows": int(mlp_chunk_rows),
            "lora_chunk_mib": int(lora_chunk_mib),
            "manual_reserve_gib": float(manual_reserve_gib),
        }
        transformer_options[CONFIG_KEY] = config.copy()

        for index, block in enumerate(diffusion_model.blocks):
            path = f"diffusion_model.blocks.{index}.mlp.forward"
            previous_forward = patched.get_model_object(path)
            patched.add_object_patch(
                path,
                _make_chunked_mlp_forward(previous_forward, config),
            )

        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "H3LongSequence_turbo_lora",
            _lora_install_wrapper(diffusion_model, config),
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            "H3LongSequence_activation_reserve",
            _activation_reserve_wrapper(config),
        )
        LOG.info(
            "H3 Long-Sequence armed: profile=%s, MLP chunk=%d rows, "
            "LoRA temporary target=%d MiB, manual reserve=%.1f GiB; short "
            "sequences are bypassed.",
            profile,
            int(mlp_chunk_rows),
            int(lora_chunk_mib),
            float(manual_reserve_gib),
        )
        return (patched,)
