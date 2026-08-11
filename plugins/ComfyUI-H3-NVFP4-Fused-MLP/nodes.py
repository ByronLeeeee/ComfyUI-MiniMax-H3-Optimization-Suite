from __future__ import annotations

import logging
import math
import types

import torch

import comfy.ops
import comfy.patcher_extension
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout

from .kernels import (
    TRITON_AVAILABLE,
    fused_swiglu_quantize_nvfp4,
    runtime_description,
)


LOG = logging.getLogger("h3_nvfp4_fused_mlp")
CONFIG_KEY = "h3_nvfp4_fused_mlp_config"
_ANNOUNCED_SHAPES: set[tuple] = set()
_ANNOUNCED_FALLBACKS: set[str] = set()
_ANNOUNCED_RESIDENCY: set[tuple] = set()
_ANNOUNCED_RESIDENCY_WARNINGS: set[str] = set()
_ANNOUNCED_POST_RESIDENCY: set[tuple] = set()


_RESIDENCY_MODES = {
    "off": (0.0, 0),
    # Only target-video/audio rows are counted, so text and reference rows are
    # already excluded before applying these additional safety margins.
    "auto_safe": (0.50, 640),
    "auto_balanced": (0.72, 896),
}


def _h3_model(model):
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, MiniMaxH3Model):
        raise ValueError(
            "H3 NVFP4 Fused MLP only supports ComfyUI's native MiniMax H3 model."
        )
    return diffusion_model


def _native_mlp(mlp, fc1_output):
    return comfy.ops.linear_input_act(mlp.fc2, fc1_output, "swiglu")


def _latent_shapes_from_conds(conds):
    if not isinstance(conds, dict):
        return None
    for cond_list in conds.values():
        for cond in cond_list or ():
            if not isinstance(cond, dict):
                continue
            # PREPARE_SAMPLING runs before process_conds(). At that point H3's
            # converted conditioning still carries latent_shapes at top level;
            # after process_conds() the same value lives under model_conds.
            latent_shapes = cond.get("latent_shapes")
            if latent_shapes is None:
                latent_shapes = cond.get("model_conds", {}).get("latent_shapes")
            if latent_shapes is not None:
                return getattr(latent_shapes, "cond", latent_shapes)
    return None


def _target_sequence_rows(latent_shapes) -> int:
    """Conservative H3 row count: target video/audio only, no text or refs."""
    if not latent_shapes or len(latent_shapes) < 2:
        return 0
    video = tuple(int(v) for v in latent_shapes[0])
    audio = tuple(int(v) for v in latent_shapes[1])
    if len(video) != 5 or len(audio) != 4:
        return 0
    batch, _, latent_t, latent_h, latent_w = video
    # H3's video patch size is (1, 2, 2); its audio rows are channel-major
    # [audio_channels, audio_time]. MiniMax H3 supports batch size one, but keep
    # batch in the video formula so malformed inputs fail conservatively.
    video_rows = batch * latent_t * math.ceil(latent_h / 2) * math.ceil(latent_w / 2)
    audio_rows = audio[-2] * audio[-1]
    return video_rows + audio_rows


def _target_sequence_rows_from_noise(noise_shape) -> int:
    """Lower-bound H3 rows from its packed target latent shape.

    PREPARE_SAMPLING currently runs before CFGGuider attaches latent_shapes.
    Native H3 packs the target modalities as [B, 1, N]. A video token contains
    24 channels x a 2x2 patch = 96 scalars, while an audio token contains only
    32. Dividing all packed scalars by 96 therefore never over-counts target
    rows, even when audio is present.
    """
    if noise_shape is None or len(noise_shape) != 3 or int(noise_shape[1]) != 1:
        return 0
    packed_values = math.prod(int(v) for v in noise_shape)
    return packed_values // 96


def _residency_reclaim_bytes(
    conds, noise_shape, ffn_width: int, mode: str
) -> tuple[int, int]:
    fraction, cap_mib = _RESIDENCY_MODES[mode]
    rows = _target_sequence_rows(_latent_shapes_from_conds(conds))
    if rows <= 0:
        rows = _target_sequence_rows_from_noise(noise_shape)
    theoretical = rows * ffn_width * 2  # H3 FC1 output is BF16/FP16.
    cap = cap_mib * 1024 * 1024
    return min(int(theoretical * fraction), cap), theoretical


def _residency_prepare_wrapper(ffn_width: int, mode: str):
    def wrapper(executor, model, noise_shape, conds, *args, **kwargs):
        reclaim, theoretical = _residency_reclaim_bytes(
            conds, noise_shape, ffn_width, mode
        )
        if reclaim < 64 * 1024 * 1024 or len(noise_shape) < 2:
            return executor(model, noise_shape, conds, *args, **kwargs)

        # PREPARE_SAMPLING consumes noise_shape only for memory estimation. The
        # returned sampler still receives the original latent. Find a synthetic
        # packed width whose estimate is lower by the safe reclaim amount; this
        # gives ComfyUI's dynamic loader the saved activation headroom for more
        # resident weights without changing model inputs or numerical results.
        try:
            import comfy.sampler_helpers

            _, base_minimum = comfy.sampler_helpers.estimate_memory(
                model, noise_shape, conds
            )
            target_minimum = max(0, base_minimum - reclaim)
            adjusted = list(noise_shape)
            original_width = int(adjusted[-1])
            low, high, best_width = 1, original_width, original_width
            best_minimum = base_minimum
            while low <= high:
                candidate_width = (low + high) // 2
                adjusted[-1] = candidate_width
                _, candidate_minimum = comfy.sampler_helpers.estimate_memory(
                    model, adjusted, conds
                )
                if candidate_minimum <= target_minimum:
                    best_width = candidate_width
                    best_minimum = candidate_minimum
                    low = candidate_width + 1
                else:
                    high = candidate_width - 1

            actual_reclaim = max(0, int(base_minimum - best_minimum))
            if best_width >= original_width or actual_reclaim < 32 * 1024 * 1024:
                return executor(model, noise_shape, conds, *args, **kwargs)
            adjusted[-1] = best_width
            announce_key = (
                mode,
                tuple(int(v) for v in noise_shape),
                round(actual_reclaim / (1024 * 1024)),
            )
            resident_before = int(model.loaded_size())
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if reason not in _ANNOUNCED_RESIDENCY_WARNINGS:
                _ANNOUNCED_RESIDENCY_WARNINGS.add(reason)
                LOG.warning(
                    "H3 NVFP4 residency hint disabled for this run: %s", reason
                )
            return executor(model, noise_shape, conds, *args, **kwargs)

        # Do not catch failures from ComfyUI's real preparation/loading call:
        # executing it twice after an OOM or interrupt would hide the cause and
        # can leave a partially changed loader state.
        result = executor(model, adjusted, conds, *args, **kwargs)
        resident_after = int(model.loaded_size())
        if announce_key not in _ANNOUNCED_RESIDENCY:
            _ANNOUNCED_RESIDENCY.add(announce_key)
            LOG.info(
                "H3 NVFP4 residency hint %s: theoretical transient saving "
                "%.1f MiB; returned %.1f MiB to ComfyUI's weight budget "
                "(estimator %.1f -> %.1f MiB; loader-tracked weights before "
                "first forward %.1f -> %.1f MiB).",
                mode,
                theoretical / (1024 * 1024),
                actual_reclaim / (1024 * 1024),
                base_minimum / (1024 * 1024),
                best_minimum / (1024 * 1024),
                resident_before / (1024 * 1024),
                resident_after / (1024 * 1024),
            )
        return result

    return wrapper


def _residency_sampler_wrapper(mode: str):
    def wrapper(
        executor, guider, sigmas, extra_args, callback, noise, *args, **kwargs
    ):
        patcher = guider.model_patcher
        resident_before = int(patcher.loaded_size())
        result = executor(
            guider, sigmas, extra_args, callback, noise, *args, **kwargs
        )
        resident_after = int(patcher.loaded_size())
        key = (mode, len(sigmas), round(resident_after / (1024 * 1024)))
        if key not in _ANNOUNCED_POST_RESIDENCY:
            _ANNOUNCED_POST_RESIDENCY.add(key)
            LOG.info(
                "H3 NVFP4 residency %s after sampling: loader-tracked resident "
                "weights %.1f -> %.1f MiB.",
                mode,
                resident_before / (1024 * 1024),
                resident_after / (1024 * 1024),
            )
        return result

    return wrapper


def _fallback(reason: str):
    if reason not in _ANNOUNCED_FALLBACKS:
        _ANNOUNCED_FALLBACKS.add(reason)
        LOG.warning("H3 NVFP4 Fused MLP falling back to native path: %s", reason)


def _can_fuse(mlp, x) -> tuple[bool, str]:
    if not TRITON_AVAILABLE:
        return False, runtime_description()
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        return False, f"unsupported activation device/dtype {x.device}/{x.dtype}"
    if getattr(mlp.fc2, "pre_quant_scale", None) is not None:
        return False, "FC2 pre_quant_scale is not supported"
    if len(getattr(mlp.fc2, "weight_function", ())) or len(
        getattr(mlp.fc2, "bias_function", ())
    ):
        return False, "FC2 has dynamic weight/bias patches (for example a LoRA)"
    weight = getattr(mlp.fc2, "weight", None)
    if not isinstance(weight, QuantizedTensor):
        return False, "FC2 weight is not a QuantizedTensor"
    if getattr(weight, "_layout_cls", None) != "TensorCoreNVFP4Layout":
        return False, f"FC2 layout is {getattr(weight, '_layout_cls', None)!r}, not NVFP4"
    if getattr(weight._params, "transposed", False):
        return False, "transposed NVFP4 FC2 weight is unsupported"
    return True, ""


def _fused_forward(mlp):
    def forward(self, x):
        fc1_output = self.fc1(x)
        supported, reason = _can_fuse(self, fc1_output)
        if not supported:
            _fallback(reason)
            return _native_mlp(self, fc1_output)

        try:
            packed, tensor_scale, block_scales, orig_shape = (
                fused_swiglu_quantize_nvfp4(
                    fc1_output,
                    precision="native_rounding",
                )
            )
            params = TensorCoreNVFP4Layout.Params(
                scale=tensor_scale,
                orig_dtype=fc1_output.dtype,
                orig_shape=orig_shape,
                block_scale=block_scales,
            )
            quantized_input = QuantizedTensor(
                packed, "TensorCoreNVFP4Layout", params
            )
            shape_key = (
                *orig_shape,
                str(fc1_output.dtype),
            )
            if shape_key not in _ANNOUNCED_SHAPES:
                _ANNOUNCED_SHAPES.add(shape_key)
                saved_mib = orig_shape[0] * orig_shape[1] * fc1_output.element_size() / 1048576
                LOG.info(
                    "H3 NVFP4 Fused MLP active for MxK=%dx%d, exact native rounding; "
                    "avoided %.1f MiB "
                    "BF16/FP16 SwiGLU intermediate per MLP call.",
                    orig_shape[0],
                    orig_shape[1],
                    saved_mib,
                )
            return self.fc2(quantized_input)
        except Exception as exc:
            _fallback(f"runtime error {type(exc).__name__}: {exc}")
            return _native_mlp(self, fc1_output)

    return types.MethodType(forward, mlp)


class H3NVFP4FusedMLP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "vram_residency": (
                    ["off", "auto_safe", "auto_balanced"],
                    {"default": "off"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/minimax_h3"
    DESCRIPTION = (
        "Fuses H3 MLP SwiGLU with FC2 input NVFP4 quantization, avoiding the "
        "large activated BF16 intermediate while retaining comfy_kitchen's native "
        "cuBLAS NVFP4 GEMM. Optional VRAM residency modes return a conservative "
        "part of that headroom to ComfyUI's dynamic weight loader. It does not "
        "skip blocks or reduce sampling steps."
    )

    def apply(self, model, vram_residency="off"):
        if not TRITON_AVAILABLE:
            raise RuntimeError(runtime_description())
        if vram_residency not in _RESIDENCY_MODES:
            raise ValueError(f"unknown VRAM residency mode: {vram_residency}")

        patched = model.clone()
        diffusion_model = _h3_model(patched)
        transformer_options = patched.model_options.setdefault(
            "transformer_options", {}
        )
        if CONFIG_KEY in transformer_options:
            raise ValueError("H3 NVFP4 Fused MLP has already been applied.")
        transformer_options[CONFIG_KEY] = {
            "mode": "exact_native_rounding",
            "vram_residency": vram_residency,
        }

        for index, block in enumerate(diffusion_model.blocks):
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.mlp.forward",
                _fused_forward(block.mlp),
            )

        if vram_residency != "off":
            fc2 = diffusion_model.blocks[0].mlp.fc2
            ffn_width = int(getattr(fc2, "in_features", 0))
            if ffn_width <= 0:
                weight = getattr(fc2, "weight", None)
                weight_shape = getattr(weight, "shape", ())
                if len(weight_shape) != 2:
                    raise ValueError("could not determine H3 FC2 input width")
                ffn_width = int(weight_shape[1])
            patched.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                "H3NVFP4FusedMLP_residency",
                _residency_prepare_wrapper(ffn_width, vram_residency),
            )
            patched.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
                "H3NVFP4FusedMLP_residency_report",
                _residency_sampler_wrapper(vram_residency),
            )

        LOG.info(
            "H3 NVFP4 Fused MLP patched %d DiT MLPs: exact native rounding, "
            "VRAM residency %s (%s).",
            len(diffusion_model.blocks),
            vram_residency,
            runtime_description(),
        )
        return (patched,)
