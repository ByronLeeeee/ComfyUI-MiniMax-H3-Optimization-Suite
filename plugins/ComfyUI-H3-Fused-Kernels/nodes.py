import logging
import types

from comfy.ldm.minimax.model import MiniMaxH3Model

from .kernels import mod_gate, norm_scale_shift, runtime_description


LOG = logging.getLogger("h3_fused_kernels")
FUSED_KEY = "h3_fused_kernels_config"


def _h3_model(model):
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, MiniMaxH3Model):
        raise ValueError("H3 Fused Kernels only supports ComfyUI's native MiniMax H3 model.")
    return diffusion_model


def _fused_forward(block, backend, precision):
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        if transformer_options is None:
            transformer_options = {}

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = norm_scale_shift(
            self.norm1,
            x,
            shift_msa,
            scale_msa,
            mod_segments,
            backend,
            precision,
        )
        attention = self.attn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        x = mod_gate(x, gate_msa, attention, mod_segments, backend)
        h = norm_scale_shift(
            self.norm2,
            x,
            shift_mlp,
            scale_mlp,
            mod_segments,
            backend,
            precision,
        )
        mlp = self.mlp(h)
        return mod_gate(x, gate_mlp, mlp, mod_segments, backend)

    return types.MethodType(forward, block)


class H3FusedKernels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (
                    ["auto", "triton", "torch_reference"],
                    {"default": "auto"},
                ),
                "precision": (
                    ["accurate", "fast"],
                    {"default": "accurate"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/minimax_h3"
    DESCRIPTION = (
        "Replaces MiniMax H3's per-segment AdaLN and gated-residual launches with "
        "whole-packed-sequence Triton kernels. It does not skip blocks or reduce steps. "
        "Accurate mode keeps native RMSNorm; fast mode can also fuse RMSNorm + AdaLN."
    )

    def apply(self, model, backend, precision):
        patched = model.clone()
        diffusion_model = _h3_model(patched)
        transformer_options = patched.model_options.setdefault("transformer_options", {})

        if FUSED_KEY in transformer_options:
            raise ValueError("H3 Fused Kernels has already been applied to this model.")

        replacements = transformer_options.get("patches_replace", {}).get("dit", {})
        if any(key[0] == "double_block" for key in replacements):
            raise ValueError(
                "H3 Fused Kernels cannot share one model chain with a node that replaces "
                "H3 double blocks (including H3 Universal MLP Controller). Use separate "
                "A/B branches while testing."
            )

        transformer_options[FUSED_KEY] = {
            "backend": backend,
            "precision": precision,
        }
        for index, block in enumerate(diffusion_model.blocks):
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.forward",
                _fused_forward(block, backend, precision),
            )

        LOG.info(
            "H3 Fused Kernels applied to %d blocks: backend=%s, precision=%s (%s).",
            len(diffusion_model.blocks),
            backend,
            precision,
            runtime_description(),
        )
        return (patched,)
