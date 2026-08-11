from __future__ import annotations

import logging

import torch

from comfy.ldm.minimax.model import MiniMaxH3Model


LOG = logging.getLogger("h3_low_step_sigmas")
PROFILE_BIAS = {
    "stock_simple": 1.0,
    "balanced_late": 1.25,
    "strong_late": 1.50,
    "custom": None,
}


class H3LowStepSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 6, "min": 2, "max": 100}),
                "profile": (
                    list(PROFILE_BIAS),
                    {"default": "stock_simple"},
                ),
                "late_bias": (
                    "FLOAT",
                    {
                        "default": 1.25,
                        "min": 0.50,
                        "max": 3.00,
                        "step": 0.05,
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/minimax_h3/optimization"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "MiniMax H3 low-step schedule. A bias above 1 moves a portion of the "
        "limited evaluations toward later/cleaner denoising while retaining "
        "the model's native video shift. stock_simple exactly reproduces "
        "ComfyUI's simple scheduler. Intended for 5-10 step experiments."
    )

    def get_sigmas(self, model, steps=6, profile="stock_simple", late_bias=1.25):
        diffusion_model = model.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("H3 Low-Step Sigma Schedule only supports native MiniMax H3")
        model_sampling = model.get_model_object("model_sampling")
        bias = PROFILE_BIAS[profile]
        if bias is None:
            bias = float(late_bias)

        # ComfyUI simple uses base times 1, 1-1/N, ..., 1/N and then maps
        # them through the model's native shift. Power-warping that base grid
        # keeps both endpoints and preserves the H3 shift/audio remapping.
        index = torch.arange(steps, dtype=torch.float64)
        if bias == 1.0:
            # Match comfy.samplers.simple_scheduler's discrete indexing, not
            # merely its idealized continuous formula.
            spacing = len(model_sampling.sigmas) / float(steps)
            sigmas = torch.tensor(
                [
                    float(model_sampling.sigmas[-(1 + int(i * spacing))])
                    for i in range(steps)
                ],
                dtype=torch.float32,
            )
        else:
            base_time = (1.0 - index / float(steps)).pow(bias)
            sigmas = model_sampling.sigma(
                base_time.to(dtype=torch.float32)
                * float(model_sampling.multiplier)
            ).detach().cpu()
        sigmas = torch.cat((sigmas, torch.zeros(1, dtype=sigmas.dtype)))
        if not bool(torch.all(sigmas[:-1] > sigmas[1:])):
            raise RuntimeError("Generated H3 sigma schedule is not strictly decreasing")
        LOG.info(
            "H3 low-step sigmas: steps=%d profile=%s bias=%.3f values=%s",
            steps,
            profile,
            bias,
            [round(float(value), 6) for value in sigmas],
        )
        return (sigmas,)
