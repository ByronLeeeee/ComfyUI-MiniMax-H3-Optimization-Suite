from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
import types
from collections import defaultdict

import torch

import comfy.patcher_extension
import folder_paths
from comfy.ldm.minimax.model import MiniMaxH3Model, _mod_gate, _mod_scale_shift


LOG = logging.getLogger("h3_step_profiler")
CONFIG_KEY = "h3_step_profiler_config"
TIMER_CONFIG_KEY = "h3_denoise_timer_config"


def _mib(value):
    return float(value) / (1024.0 * 1024.0)


class _ProfileState:
    def __init__(self):
        self.active = False
        self.call_index = 0
        self.events = defaultdict(list)

    def reset_events(self):
        self.events.clear()

    def timed(self, category, fn):
        if not self.active:
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        self.events[category].append((start, end))
        return result

    def elapsed_ms(self, category):
        return sum(start.elapsed_time(end) for start, end in self.events[category])


def _timed_forward(module, original_forward, state, category):
    def forward(self, *args, **kwargs):
        return state.timed(category, lambda: original_forward(*args, **kwargs))

    return types.MethodType(forward, module)


def _profiled_block_forward(block, state):
    def forward(
        self, x, t_emb, mod_segments, rope_freqs, transformer_options={}
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_proj(t_emb)
        )
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        attn = state.timed(
            "attention_total",
            lambda: self.attn(
                h,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            ),
        )
        x = _mod_gate(x, gate_msa, attn, mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        mlp = state.timed("mlp_total", lambda: self.mlp(h))
        return _mod_gate(x, gate_mlp, mlp, mod_segments)

    return types.MethodType(forward, block)


def _device_time_us(event):
    for name in (
        "self_device_time_total",
        "self_cuda_time_total",
        "device_time_total",
        "cuda_time_total",
    ):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _profiler_rows(profiler):
    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        rows.append(
            {
                "name": event.key,
                "count": int(event.count),
                "self_device_ms": _device_time_us(event) / 1000.0,
                "cpu_total_ms": float(getattr(event, "cpu_time_total", 0.0))
                / 1000.0,
                "input_shapes": str(getattr(event, "input_shapes", "")),
            }
        )
    rows.sort(key=lambda row: row["self_device_ms"], reverse=True)
    return rows


def _raw_cuda_activity(profiler):
    h2d_ms = 0.0
    d2h_ms = 0.0
    memcpy_ms = 0.0
    kernel_ms = 0.0
    cuda_events = 0
    names = defaultdict(lambda: {"count": 0, "device_ms": 0.0})
    for event in profiler.events():
        device_type = str(getattr(event, "device_type", "")).lower()
        if "cuda" not in device_type:
            continue
        cuda_events += 1
        name = str(getattr(event, "name", ""))
        lower = name.lower()
        duration_ms = _device_time_us(event) / 1000.0
        names[name]["count"] += 1
        names[name]["device_ms"] += duration_ms
        if "memcpy" in lower or "memset" in lower:
            memcpy_ms += duration_ms
            if "htod" in lower or "host to device" in lower:
                h2d_ms += duration_ms
            elif "dtoh" in lower or "device to host" in lower:
                d2h_ms += duration_ms
        else:
            kernel_ms += duration_ms
    top = [
        {"name": name, **values}
        for name, values in sorted(
            names.items(), key=lambda item: item[1]["device_ms"], reverse=True
        )[:100]
    ]
    return {
        "cuda_event_count": cuda_events,
        "kernel_activity_ms": kernel_ms,
        "memcpy_activity_ms": memcpy_ms,
        "h2d_activity_ms": h2d_ms,
        "d2h_activity_ms": d2h_ms,
        "top_cuda_events": top,
    }


def _predict_noise_profiler(state, profile_call, output_prefix):
    def wrapper(executor, x, timestep, model_options={}, seed=None):
        state.call_index += 1
        if state.call_index != profile_call:
            return executor(x, timestep, model_options, seed)

        output_dir = os.path.join(folder_paths.get_output_directory(), "profiles")
        os.makedirs(output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{output_prefix}_{stamp}_call{profile_call}"
        trace_path = os.path.join(output_dir, base + ".json")
        summary_path = os.path.join(output_dir, base + "_summary.json")

        torch.cuda.synchronize()
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        state.reset_events()
        LOG.info("H3 profiler capturing denoise call %d", profile_call)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            state.active = True
            total_start.record()
            try:
                result = executor(x, timestep, model_options, seed)
            finally:
                total_end.record()
                state.active = False
                torch.cuda.synchronize()

        profiler.export_chrome_trace(trace_path)
        total_ms = total_start.elapsed_time(total_end)
        attention_ms = state.elapsed_ms("attention_total")
        mlp_ms = state.elapsed_ms("mlp_total")
        qkv_ms = state.elapsed_ms("attention_qkv_linear")
        out_proj_ms = state.elapsed_ms("attention_out_linear")
        fc1_ms = state.elapsed_ms("mlp_fc1_linear")
        fc2_ms = state.elapsed_ms("mlp_fc2_linear")
        summary = {
            "profile_call": profile_call,
            "total_step_ms": total_ms,
            "critical_path": {
                "attention_total_ms": attention_ms,
                "attention_qkv_linear_ms": qkv_ms,
                "attention_out_linear_ms": out_proj_ms,
                "attention_core_and_rope_ms": max(0.0, attention_ms - qkv_ms - out_proj_ms),
                "mlp_total_ms": mlp_ms,
                "mlp_fc1_linear_ms": fc1_ms,
                "mlp_fc2_linear_ms": fc2_ms,
                "mlp_activation_quantize_ms": max(0.0, mlp_ms - fc1_ms - fc2_ms),
                "other_ms": max(0.0, total_ms - attention_ms - mlp_ms),
            },
            "cuda_activity": _raw_cuda_activity(profiler),
            "top_operators": _profiler_rows(profiler)[:250],
            "trace_path": trace_path,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        LOG.info(
            "H3 profile saved: total %.1f ms; attention %.1f ms; MLP %.1f ms; "
            "other %.1f ms; %s",
            total_ms,
            attention_ms,
            mlp_ms,
            summary["critical_path"]["other_ms"],
            summary_path,
        )
        return result

    return wrapper


class H3StepProfiler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "profile_call": ("INT", {"default": 2, "min": 1, "max": 100}),
                "output_prefix": (
                    "STRING",
                    {"default": "h3_step", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/minimax_h3/diagnostic"
    DESCRIPTION = (
        "Profiles one H3 denoise call with CUDA events and torch.profiler. "
        "Diagnostic only; the captured call is slower and a trace is written "
        "under output/profiles."
    )

    def apply(self, model, profile_call=2, output_prefix="h3_step"):
        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("H3 Single-Step Profiler only supports native MiniMax H3")
        options = patched.model_options.setdefault("transformer_options", {})
        if CONFIG_KEY in options:
            raise ValueError("H3 Single-Step Profiler has already been applied")
        options[CONFIG_KEY] = {
            "profile_call": int(profile_call),
            "output_prefix": output_prefix,
        }

        state = _ProfileState()
        for index, block in enumerate(diffusion_model.blocks):
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.forward",
                _profiled_block_forward(block, state),
            )
            for name, category in (
                ("attn.qkv_proj", "attention_qkv_linear"),
                ("attn.out_proj", "attention_out_linear"),
                ("mlp.fc1", "mlp_fc1_linear"),
                ("mlp.fc2", "mlp_fc2_linear"),
            ):
                module = block.get_submodule(name)
                patched.add_object_patch(
                    f"diffusion_model.blocks.{index}.{name}.forward",
                    _timed_forward(module, module.forward, state, category),
                )

        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
            "H3StepProfiler_predict_noise",
            _predict_noise_profiler(state, int(profile_call), output_prefix),
        )
        LOG.info(
            "H3 Single-Step Profiler armed for denoise call %d (%d blocks)",
            profile_call,
            len(diffusion_model.blocks),
        )
        return (patched,)


def _denoise_timer(output_prefix, expected_calls):
    call_index = 0
    samples_ms = []
    memory_samples = []

    def wrapper(executor, x, timestep, model_options={}, seed=None):
        nonlocal call_index
        call_index += 1
        torch.cuda.synchronize()
        device = x.device if isinstance(x, torch.Tensor) else torch.cuda.current_device()
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = executor(x, timestep, model_options, seed)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        samples_ms.append(elapsed)
        memory_sample = {
            "allocated_before_mib": _mib(allocated_before),
            "allocated_after_mib": _mib(torch.cuda.memory_allocated(device)),
            "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
            "reserved_before_mib": _mib(reserved_before),
            "reserved_after_mib": _mib(torch.cuda.memory_reserved(device)),
            "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved(device)),
        }
        memory_sample["incremental_peak_allocated_mib"] = max(
            0.0,
            memory_sample["peak_allocated_mib"] - memory_sample["allocated_before_mib"],
        )
        memory_samples.append(memory_sample)
        LOG.info(
            "H3 denoise timer call %d/%d: %.3f ms; peak %.1f MiB "
            "(+%.1f MiB over call start)",
            call_index,
            expected_calls,
            elapsed,
            memory_sample["peak_allocated_mib"],
            memory_sample["incremental_peak_allocated_mib"],
        )
        if call_index == expected_calls:
            output_dir = os.path.join(folder_paths.get_output_directory(), "profiles")
            os.makedirs(output_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(output_dir, f"{output_prefix}_{stamp}_timing.json")
            payload = {
                "expected_calls": expected_calls,
                "observed_calls": call_index,
                "samples_ms": samples_ms,
                "memory_samples": memory_samples,
                "total_ms": sum(samples_ms),
                "mean_ms": sum(samples_ms) / len(samples_ms),
                "steady_mean_excluding_first_ms": (
                    sum(samples_ms[1:]) / len(samples_ms[1:])
                    if len(samples_ms) > 1
                    else samples_ms[0]
                ),
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            LOG.info("H3 denoise timing saved: %s", path)
        return result

    return wrapper


class H3DenoiseTimer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "expected_calls": ("INT", {"default": 2, "min": 1, "max": 200}),
                "output_prefix": (
                    "STRING",
                    {"default": "h3_timing", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/minimax_h3/diagnostic"
    DESCRIPTION = (
        "Lightweight CUDA-event timing for each H3 denoise call. It synchronizes "
        "at call boundaries and writes JSON under output/profiles."
    )

    def apply(self, model, expected_calls=2, output_prefix="h3_timing"):
        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("H3 Denoise Timer only supports native MiniMax H3")
        options = patched.model_options.setdefault("transformer_options", {})
        if TIMER_CONFIG_KEY in options:
            raise ValueError("H3 Denoise Timer has already been applied")
        options[TIMER_CONFIG_KEY] = {
            "expected_calls": int(expected_calls),
            "output_prefix": output_prefix,
        }
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
            "H3DenoiseTimer_predict_noise",
            _denoise_timer(output_prefix, int(expected_calls)),
        )
        return (patched,)


class H3BenchmarkLatentSink:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ()
    FUNCTION = "consume"
    OUTPUT_NODE = True
    CATEGORY = "sampling/minimax_h3/diagnostic"
    DESCRIPTION = "Consumes H3 latent output without invoking either VAE decoder."

    def consume(self, samples):
        return ()


def _fingerprint_value(value, digest, tensors, path="root"):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous()
        raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
        metadata = {
            "path": path,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(value.device),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        digest.update(path.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(raw)
        tensors.append(metadata)
        return
    nested = getattr(value, "tensors", None)
    if isinstance(nested, (list, tuple)):
        for index, item in enumerate(nested):
            _fingerprint_value(item, digest, tensors, f"{path}.tensors[{index}]")
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            _fingerprint_value(value[key], digest, tensors, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _fingerprint_value(item, digest, tensors, f"{path}[{index}]")


class H3LatentFingerprintSink:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "output_prefix": (
                    "STRING",
                    {"default": "h3_latent", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "consume"
    OUTPUT_NODE = True
    CATEGORY = "sampling/minimax_h3/diagnostic"
    DESCRIPTION = (
        "Consumes an H3 latent without VAE decoding and writes exact per-tensor "
        "plus combined SHA-256 fingerprints under output/profiles."
    )

    def consume(self, samples, output_prefix="h3_latent"):
        digest = hashlib.sha256()
        tensors = []
        _fingerprint_value(samples, digest, tensors)
        if not tensors:
            raise ValueError("No tensors were found in the LATENT input")
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_prefix).strip("._")
        safe_prefix = safe_prefix or "h3_latent"
        output_dir = os.path.join(folder_paths.get_output_directory(), "profiles")
        os.makedirs(output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(output_dir, f"{safe_prefix}_{stamp}_fingerprint.json")
        payload = {
            "combined_sha256": digest.hexdigest(),
            "tensor_count": len(tensors),
            "tensors": tensors,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        LOG.info("H3 latent fingerprint saved: %s (%s)", path, digest.hexdigest())
        return ()
