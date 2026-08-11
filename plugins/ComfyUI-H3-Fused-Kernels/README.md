# ComfyUI H3 Fused Kernels

An independent experimental optimization node for ComfyUI's native MiniMax H3
implementation.

This is not the suite's **H3 NVFP4 Fused MLP** node. Fused MLP targets the
SwiGLU-to-FC2 NVFP4 activation/quantization boundary and is Blackwell/NVFP4
specific. H3 Fused Kernels targets per-segment AdaLN modulation and gated
residual launches in the surrounding H3 block.

## What it changes

Native H3 applies AdaLN modulation and gated residuals separately for each packed
text/image/video/audio segment in every DiT block. This node keeps the same 50
blocks, attention, MLP and sampling steps, but maps every token to its modulation
row once and launches kernels across the complete packed sequence.

- `accurate`: keeps ComfyUI/PyTorch RMSNorm and keeps the multiply/add rounding
  boundary. It fuses segments, not denoising work.
- `fast`: also combines scale and shift and, when the RMSNorm weight is already on
  the CUDA device with no runtime weight function, fuses RMSNorm + AdaLN.
- `torch_reference`: executes the native PyTorch math through the same patched
  block-forward path. Use it to measure the node's fixed patch overhead.
- `auto`: tries Triton on compatible NVIDIA CUDA tensors and permanently falls
  back to PyTorch for the session if first-use compilation fails.

No block is skipped and no result from a previous denoising step is reused.
`accurate` is the recommended first test. Floating-point kernels are not promised
to be bit-identical across backends, so quality still needs an A/B check.

The measured effect depends on resolution and sequence shape. Current suite
results confirm that `auto + accurate` works in the 15-second Low-Memory Sage2
and Long-Sequence chain, but do not establish a universal speed improvement.

## Placement

```text
Load MiniMax H3 model
-> LoRA/model patches (optional)
-> H3 Fused Kernels
-> Patch Sage Attention (optional)
-> guider / sampler
```

Do not place H3 Fused Kernels and H3 Universal MLP Controller on the same model
chain during this experiment. Both plugins can remain installed, and separate
workflow branches can test them independently.

## Controlled benchmark

Keep prompt, seed, sampler, 20-step schedule, resolution, duration and reference
input identical. Run in this order:

1. Node bypassed.
2. `torch_reference`.
3. `auto + accurate` twice.
4. Optionally `auto + fast` twice.

Discard the first Triton timing because kernels compile on first use. Compare the
sampling progress-bar duration separately from VAE/video/audio encoding time.

## Compatibility

- Native ComfyUI MiniMax H3 only.
- Triton path requires a working Triton build and CUDA GPU.
- CPU, unsupported dtypes and failed `auto` compilation use PyTorch fallback.
- SageAttention is compatible because this node does not replace the attention
  backend.

## Installation

Copy `ComfyUI-H3-Fused-Kernels` into `ComfyUI/custom_nodes`, then restart ComfyUI.
Triton is optional for loading the plugin, but required for acceleration.
