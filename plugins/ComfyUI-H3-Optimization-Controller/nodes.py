from __future__ import annotations

import importlib
import logging

import comfy.samplers


LOG = logging.getLogger("h3_optimization_controller")

CONTROLLER_KEY = "h3_optimization_controller_config"
FUSED_KEY = "h3_nvfp4_fused_mlp_config"
LOW_MEMORY_KEY = "h3_low_memory_sage2_config"
ATTENTION_OVERRIDE_KEY = "optimized_attention_override"

PRESETS = [
    "exact_speed",
    "exact_low_vram",
    "off",
]

SAMPLERS = ["CAB-2", "CAB-3", "res_multistep"]
SIGMA_PROFILES = ["stock_simple", "balanced_late", "strong_late", "custom"]


def _node_class(node_id: str):
    # Custom nodes are all registered before a graph can execute, so resolving
    # here avoids any dependency on custom-node import order.
    comfy_nodes = importlib.import_module("nodes")
    node_class = comfy_nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if node_class is None:
        raise RuntimeError(
            f"Required H3 component '{node_id}' is not installed or failed to load. "
            "Install/enable the matching standalone H3 optimization plugin."
        )
    return node_class


def _invoke(node_id: str, function_name: str, **kwargs):
    instance = _node_class(node_id)()
    function = getattr(instance, function_name, None)
    if function is None:
        raise RuntimeError(
            f"H3 component '{node_id}' does not provide '{function_name}'. "
            "Update the standalone component and this controller together."
        )
    result = function(**kwargs)
    if not isinstance(result, tuple) or not result:
        raise RuntimeError(f"H3 component '{node_id}' returned an invalid result")
    return result


def _transformer_options(model):
    model_options = getattr(model, "model_options", {})
    return model_options.get("transformer_options", {})


def _ensure_attention_is_unpatched(model):
    options = _transformer_options(model)
    conflicts = [
        key
        for key in (LOW_MEMORY_KEY, ATTENTION_OVERRIDE_KEY)
        if key in options
    ]
    if conflicts:
        raise ValueError(
            "H3 Optimization Controller must receive a model before any attention "
            f"patch. Conflicting transformer options: {', '.join(conflicts)}"
        )


class H3OptimizationController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (PRESETS, {"default": "exact_speed"}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 200}),
                "fused_mlp": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vram_residency": (
                    ["off", "auto_safe", "auto_balanced"],
                    {"default": "off"},
                ),
                "allow_compile": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "INT")
    RETURN_NAMES = ("model", "steps")
    FUNCTION = "apply"
    CATEGORY = "sampling/minimax_h3/suite"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "One front door for the standalone H3 optimization plugins. It can "
        "combine exact fused NVFP4 MLP with stock Sage2 or exact low-memory "
        "Sage2. The steps output should "
        "be connected to H3 Optimized Sampling so both nodes stay synchronized."
    )

    def apply(
        self,
        model,
        preset="exact_speed",
        steps=10,
        fused_mlp=True,
        vram_residency="off",
        allow_compile=True,
    ):
        steps = int(steps)
        if preset not in PRESETS:
            raise ValueError(f"Unknown H3 optimization preset: {preset}")
        if preset == "off":
            LOG.info("H3 Optimization Controller: off, model left unchanged")
            return (model, steps)

        initial_options = _transformer_options(model)
        if CONTROLLER_KEY in initial_options:
            raise ValueError("H3 Optimization Controller has already been applied")
        _ensure_attention_is_unpatched(model)

        patched = model
        fused_was_present = FUSED_KEY in initial_options
        if bool(fused_mlp) and not fused_was_present:
            patched = _invoke(
                "H3NVFP4FusedMLP",
                "apply",
                model=patched,
                vram_residency=vram_residency,
            )[0]

        if preset == "exact_low_vram":
            patched = _invoke(
                "H3LowMemorySage2Attention",
                "apply",
                model=patched,
            )[0]
        elif preset == "exact_speed":
            patched = _invoke(
                "PathchSageAttentionKJ",
                "patch",
                model=patched,
                sage_attention="auto",
                allow_compile=bool(allow_compile),
            )[0]
        else:
            raise AssertionError(f"Unhandled H3 optimization preset: {preset}")

        final_options = patched.model_options.setdefault("transformer_options", {})
        final_options[CONTROLLER_KEY] = {
            "preset": preset,
            "steps": steps,
            "fused_mlp": bool(fused_mlp) or fused_was_present,
            "vram_residency": vram_residency,
            "allow_compile": bool(allow_compile),
        }
        LOG.info(
            "H3 Optimization Controller armed: preset=%s steps=%d fused_mlp=%s",
            preset,
            steps,
            bool(fused_mlp) or fused_was_present,
        )
        return (patched, steps)


class H3OptimizedSampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 10, "min": 2, "max": 100}),
                "sampler_mode": (SAMPLERS, {"default": "CAB-2"}),
                "sigma_profile": (SIGMA_PROFILES, {"default": "stock_simple"}),
                "theta": (
                    "FLOAT",
                    {"default": 0.20, "min": 0.0, "max": 1.5, "step": 0.05},
                ),
                "late_bias": (
                    "FLOAT",
                    {"default": 1.25, "min": 0.50, "max": 3.00, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER", "SIGMAS")
    RETURN_NAMES = ("sampler", "sigmas")
    FUNCTION = "build"
    CATEGORY = "sampling/minimax_h3/suite"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Builds a synchronized sampler and MiniMax H3 sigma schedule. CAB-2 "
        "with theta 0.20 and stock_simple is the validated low-step default. "
        "res_multistep is provided as the stock control."
    )

    def build(
        self,
        model,
        steps=10,
        sampler_mode="CAB-2",
        sigma_profile="stock_simple",
        theta=0.20,
        late_bias=1.25,
    ):
        if sampler_mode not in SAMPLERS:
            raise ValueError(f"Unknown H3 sampler mode: {sampler_mode}")
        if sigma_profile not in SIGMA_PROFILES:
            raise ValueError(f"Unknown H3 sigma profile: {sigma_profile}")

        if sampler_mode in ("CAB-2", "CAB-3"):
            sampler = _invoke(
                "H3CABSampler",
                "get_sampler",
                order=sampler_mode,
                theta=float(theta),
            )[0]
        else:
            sampler = comfy.samplers.sampler_object("res_multistep")

        sigmas = _invoke(
            "H3LowStepSigmas",
            "get_sigmas",
            model=model,
            steps=int(steps),
            profile=sigma_profile,
            late_bias=float(late_bias),
        )[0]
        LOG.info(
            "H3 Optimized Sampling built: sampler=%s steps=%d sigmas=%s theta=%.3f",
            sampler_mode,
            int(steps),
            sigma_profile,
            float(theta),
        )
        return (sampler, sigmas)
