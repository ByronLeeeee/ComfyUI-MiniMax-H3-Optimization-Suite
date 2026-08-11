from __future__ import annotations

import logging
import threading

import torch

import comfy.patcher_extension
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ldm.modules.attention import attention_pytorch, wrap_attn


LOG = logging.getLogger("h3_blackwell_hybrid_attention")
CONFIG_KEY = "h3_blackwell_hybrid_attention_config"


class _ScheduleState:
    def __init__(self, strategy: str, expected_calls: int, edge_calls: int):
        self.strategy = strategy
        self.expected_calls = expected_calls
        self.edge_calls = min(edge_calls, expected_calls // 2)
        self.call_index = 0
        self.use_sage3 = strategy == "sage3_all"
        self._lock = threading.Lock()

    def begin_call(self):
        with self._lock:
            if self.call_index >= self.expected_calls:
                self.call_index = 0
            self.call_index += 1
            if self.strategy == "sage3_all":
                self.use_sage3 = True
            elif self.strategy == "sage2_all":
                self.use_sage3 = False
            else:
                last_sage3_call = self.expected_calls - self.edge_calls
                self.use_sage3 = (
                    self.call_index > self.edge_calls
                    and self.call_index <= last_sage3_call
                )
            return self.call_index, self.use_sage3


def _schedule_wrapper(state: _ScheduleState):
    def wrapper(executor, x, timestep, model_options={}, seed=None):
        call_index, use_sage3 = state.begin_call()
        LOG.info(
            "H3 hybrid attention call %d/%d: %s",
            call_index,
            state.expected_calls,
            "Sage3 FP4" if use_sage3 else "Sage2++",
        )
        return executor(x, timestep, model_options, seed)

    return wrapper


def _attention_function(state: _ScheduleState, per_block_mean: bool):
    from sageattention import sageattn
    from sageattn3 import sageattn3_blackwell

    def backend(q, k, v, tensor_layout, mask):
        if not state.use_sage3:
            return sageattn(
                q,
                k,
                v,
                tensor_layout=tensor_layout,
                is_causal=False,
                attn_mask=mask,
            )

        if mask is not None:
            # SageAttention3 accepts a mask, but keeping the fallback explicit
            # avoids silent layout/broadcast differences on future Comfy builds.
            return sageattn(
                q,
                k,
                v,
                tensor_layout=tensor_layout,
                is_causal=False,
                attn_mask=mask,
            )
        if tensor_layout == "NHD":
            q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        out = sageattn3_blackwell(
            q,
            k,
            v,
            is_causal=False,
            per_block_mean=per_block_mean,
        )
        return out.transpose(1, 2) if tensor_layout == "NHD" else out

    # The selected backend changes between denoise calls. Keep this tiny Python
    # dispatch outside Dynamo/Inductor; otherwise each branch causes a costly
    # first-use graph compilation and defeats short-run acceleration.
    backend = torch.compiler.disable()(backend)

    @wrap_attn
    def attention_hybrid(
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        if kwargs.get("low_precision_attention", True) is False:
            return attention_pytorch(
                q,
                k,
                v,
                heads,
                mask=mask,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        input_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.half(), k.half(), v.half()
        if skip_reshape:
            batch, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            batch, _, hidden = q.shape
            dim_head = hidden // heads
            q, k, v = (
                tensor.view(batch, -1, heads, dim_head)
                for tensor in (q, k, v)
            )
            tensor_layout = "NHD"
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        out = backend(q, k, v, tensor_layout, mask).to(input_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
        elif skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(batch, -1, heads * dim_head)
        return out

    return attention_hybrid


class H3BlackwellHybridAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strategy": (
                    ["hybrid_sage2_edges", "sage3_all", "sage2_all"],
                    {"default": "hybrid_sage2_edges"},
                ),
                "expected_denoise_calls": (
                    "INT",
                    {"default": 10, "min": 1, "max": 200},
                ),
                "sage2_edge_calls": (
                    "INT",
                    {"default": 1, "min": 0, "max": 50},
                ),
                "sage3_per_block_mean": (
                    "BOOLEAN",
                    {"default": False},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/minimax_h3/optimization"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Blackwell-only MiniMax H3 attention controller. The hybrid strategy "
        "uses accurate Sage2++ at the first/last denoise calls and faster "
        "SageAttention3 FP4 in the middle. Set expected calls to the sampler's "
        "actual model evaluations (normally equal to steps for res_multistep)."
    )

    def apply(
        self,
        model,
        strategy="hybrid_sage2_edges",
        expected_denoise_calls=10,
        sage2_edge_calls=1,
        sage3_per_block_mean=False,
    ):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12:
            raise ValueError("H3 Blackwell Hybrid Attention requires an NVIDIA SM 12.x GPU")
        try:
            import sageattention  # noqa: F401
            import sageattn3  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "Both sageattention 2.2 and sageattn3 are required. Install a "
                "wheel matching this ComfyUI Python/Torch/CUDA environment."
            ) from error

        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("H3 Blackwell Hybrid Attention only supports native MiniMax H3")
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        if CONFIG_KEY in transformer_options:
            raise ValueError("H3 Blackwell Hybrid Attention has already been applied")

        state = _ScheduleState(
            strategy,
            int(expected_denoise_calls),
            int(sage2_edge_calls),
        )
        new_attention = _attention_function(state, bool(sage3_per_block_mean))

        def attention_override(func, *args, **kwargs):
            return new_attention.__wrapped__(*args, **kwargs)

        transformer_options[CONFIG_KEY] = {
            "strategy": strategy,
            "expected_denoise_calls": int(expected_denoise_calls),
            "sage2_edge_calls": state.edge_calls,
            "sage3_per_block_mean": bool(sage3_per_block_mean),
        }
        transformer_options["optimized_attention_override"] = attention_override
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
            "H3BlackwellHybridAttention_schedule",
            _schedule_wrapper(state),
        )
        LOG.info(
            "H3 Blackwell Hybrid Attention armed: strategy=%s, calls=%d, "
            "Sage2 edge calls=%d, per_block_mean=%s",
            strategy,
            expected_denoise_calls,
            state.edge_calls,
            sage3_per_block_mean,
        )
        return (patched,)
