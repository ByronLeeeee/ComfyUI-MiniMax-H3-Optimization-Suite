from __future__ import annotations

import logging
import types

import torch

import comfy.model_management
import comfy.quant_ops
from comfy.ldm.minimax.model import MiniMaxH3Model


LOG = logging.getLogger("h3_low_memory_sage2")
CONFIG_KEY = "h3_low_memory_sage2_config"


def _cuda_version():
    value = torch.version.cuda or "0.0"
    fields = value.split(".")
    return tuple(int(field) for field in fields[:2])


def _sage2_plan(device):
    major, minor = torch.cuda.get_device_capability(device)
    architecture = f"sm{major}{minor}"
    cuda_version = _cuda_version()
    if architecture == "sm89":
        quant_granularity = "per_thread"
        accumulation = "fp32+fp16" if cuda_version >= (12, 8) else "fp32+fp32"
    elif architecture in {"sm100", "sm120", "sm121"}:
        quant_granularity = "per_warp"
        accumulation = "fp32+fp16" if cuda_version >= (12, 8) else "fp32"
    else:
        raise ValueError(
            "H3 Low-Memory Sage2 supports Ada SM 8.9 and Blackwell "
            "SM 10.0/12.x GPUs, matching SageAttention2's FP8 CUDA path"
        )
    return architecture, quant_granularity, accumulation


def _quantize_qk(q, k, km, quant_granularity):
    from sageattention import core as sage_core
    from sageattention.quant import per_warp_int8

    if quant_granularity == "per_warp":
        return per_warp_int8(
            q,
            k,
            km,
            tensor_layout="HND",
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
        )
    return sage_core.per_thread_int8_triton(
        q,
        k,
        km,
        tensor_layout="HND",
        BLKQ=128,
        WARPQ=32,
        BLKK=64,
        WARPK=64,
    )


def _run_kernel(
    q_int8,
    k_int8,
    v_fp8,
    output,
    q_scale,
    k_scale,
    v_scale,
    accumulation,
    softmax_scale,
    quant_granularity,
):
    from sageattention import core as sage_core

    arguments = (
        q_int8,
        k_int8,
        v_fp8,
        output,
        q_scale,
        k_scale,
        v_scale,
        1,  # HND
        0,  # non-causal
        2 if quant_granularity == "per_warp" else 3,
        softmax_scale,
        0,  # do not return log-sum-exp
    )
    if accumulation == "fp32+fp16":
        sage_core._qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            *arguments
        )
    elif accumulation == "fp32+fp32":
        sage_core._qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
            *arguments
        )
    else:
        sage_core._qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(
            *arguments
        )


def _low_memory_attention(attention, qkv, rope_freqs, quant_granularity, accumulation):
    """Run stock Sage2 math while shortening the lifetime of the full QKV buffer."""
    from sageattention.quant import per_channel_fp8

    sequence = qkv.shape[0]
    heads = attention.heads
    head_dim = attention.head_dim
    inner = heads * head_dim
    if qkv.ndim != 2 or qkv.shape[-1] != inner * 3 or head_dim != 128:
        raise ValueError(
            "Unexpected MiniMax H3 QKV shape; low-memory Sage2 currently "
            "requires native 128-dimensional attention heads"
        )
    if qkv.dtype not in (torch.float16, torch.bfloat16) or not qkv.is_cuda:
        raise ValueError("H3 Low-Memory Sage2 requires CUDA FP16/BF16 activations")

    q, k, v = qkv.split(inner, dim=-1)
    v = v.view(sequence, heads, head_dim)
    if rope_freqs is not None:
        q = q.view(1, sequence, heads, head_dim)
        k = k.view(1, sequence, heads, head_dim)
        q_weight = comfy.model_management.cast_to(
            attention.q_norm.weight, device=qkv.device
        )
        k_weight = comfy.model_management.cast_to(
            attention.k_norm.weight, device=qkv.device
        )
        rotation_dimension = rope_freqs.shape[-3] * 2
        comfy.quant_ops.ck.rms_rope_split_half_(
            q,
            k,
            rope_freqs,
            q_weight,
            k_weight,
            epsilon=attention.q_norm.eps,
            rot_dim=rotation_dimension,
        )
        q, k = q[0], k[0]
    else:
        q = attention.q_norm(q.view(sequence, heads, head_dim))
        k = attention.k_norm(k.view(sequence, heads, head_dim))

    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)

    # This reproduces stock Sage2 smooth-K and Q/K quantization exactly.
    k_mean = k.mean(dim=2, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = _quantize_qk(
        q, k, k_mean, quant_granularity
    )

    # The compact V clone is one third the size of QKV. Once it exists, all
    # views of the full projection can die before Sage2 allocates V scratch and
    # the BF16 output. CUDA's caching allocator preserves stream ordering.
    v_compact = v.clone(memory_format=torch.contiguous_format)
    output_shape = tuple(q.shape)
    output_dtype = q.dtype
    output_device = q.device
    del q, k, v, k_mean, qkv

    scale_max = 2.25 if accumulation == "fp32+fp16" else 448.0
    v_fp8, v_scale, _ = per_channel_fp8(
        v_compact,
        tensor_layout="HND",
        scale_max=scale_max,
        smooth_v=False,
    )
    del v_compact

    output = torch.empty(output_shape, dtype=output_dtype, device=output_device)
    _run_kernel(
        q_int8,
        k_int8,
        v_fp8,
        output,
        q_scale,
        k_scale,
        v_scale,
        accumulation,
        head_dim**-0.5,
        quant_granularity,
    )
    return output.transpose(1, 2).reshape(sequence, inner)


def _low_memory_forward(attention, quant_granularity, accumulation):
    def forward(self, x, rope_freqs=None, transformer_options={}):
        # Keep the projection result as a temporary expression: binding it in
        # this frame would extend the full QKV allocation through the helper.
        return self.out_proj(
            _low_memory_attention(
                self,
                self.qkv_proj(x),
                rope_freqs,
                quant_granularity,
                accumulation,
            )
        )

    return types.MethodType(forward, attention)


class H3LowMemorySage2Attention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/minimax_h3/optimization"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Memory-optimized native MiniMax H3 SageAttention2. It quantizes Q/K, "
        "keeps only a compact V copy, and releases the full QKV projection "
        "before V quantization and attention output allocation. Do not combine "
        "with Patch Sage Attention or another attention replacement."
    )

    def apply(self, model):
        if not torch.cuda.is_available():
            raise ValueError("H3 Low-Memory Sage2 requires an NVIDIA CUDA GPU")
        try:
            import sageattention  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "sageattention 2.2 is required; install a wheel matching this "
                "ComfyUI Python/Torch/CUDA environment"
            ) from error

        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("H3 Low-Memory Sage2 only supports native MiniMax H3")
        options = patched.model_options.setdefault("transformer_options", {})
        if CONFIG_KEY in options:
            raise ValueError("H3 Low-Memory Sage2 has already been applied")
        if "optimized_attention_override" in options:
            raise ValueError(
                "Remove Patch Sage Attention / Hybrid Attention before applying "
                "H3 Low-Memory Sage2"
            )

        device = torch.cuda.current_device()
        architecture, quant_granularity, accumulation = _sage2_plan(device)
        options[CONFIG_KEY] = {
            "architecture": architecture,
            "qk_quant_granularity": quant_granularity,
            "pv_accumulation": accumulation,
            "compact_v": True,
        }
        for index, block in enumerate(diffusion_model.blocks):
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.attn.forward",
                _low_memory_forward(
                    block.attn,
                    quant_granularity,
                    accumulation,
                ),
            )
        LOG.info(
            "H3 Low-Memory Sage2 armed on %s: %d blocks, Q/K %s, PV %s",
            architecture,
            len(diffusion_model.blocks),
            quant_granularity,
            accumulation,
        )
        return (patched,)
