# ComfyUI H3 Optimization Controller

This plugin is a lightweight facade over the standalone MiniMax H3 optimization
plugins. It does not duplicate their CUDA/Triton kernels, and it does not modify
or replace the older experimental MLP-skip plugin.

## Nodes

### H3 Optimization Controller

Connect the native MiniMax H3 `MODEL` directly from the loader. Connect both its
`model` and `steps` outputs to **H3 Optimized Sampling**; use the patched model for
the guider as well.

Presets:

- `exact_low_vram`: fused NVFP4 MLP + exact low-memory Sage2. Saves about 567 MiB
  peak attention allocation in the validated workload, with a small speed cost.
- `exact_speed`: fused NVFP4 MLP + KJ Sage2 `auto`. Best exact-speed starting
  point; requires ComfyUI-KJNodes.
- `off`: leaves the model unchanged and only passes the step count through.

`fused_mlp` may be disabled independently. `vram_residency` controls the fused
MLP plugin's conservative attempt to return saved activation headroom to
ComfyUI's dynamic weight loader. Leave it `off` unless measuring residency.

Do not put KJ Sage Attention or Low-Memory Sage2 before this
node. The controller rejects competing attention overrides instead of silently
replacing one.

### H3 Optimized Sampling

Outputs both `SAMPLER` and `SIGMAS` for `SamplerCustomAdvanced`.

Recommended low-step starting point:

- sampler: `CAB-2`
- theta: `0.20`
- sigmas: `stock_simple`
- steps: begin at `14`, then compare `12` and `10` against a fixed 20-step
  reference. Six steps are useful only for aggressive solver research and are
  not presented as a general quality preset.

`CAB-3` and biased sigma profiles remain experimental. `res_multistep` is kept
as a convenient control. CAB changes the numerical solver; it does not skip a
model evaluation inside a nominal step.

## Required standalone plugins

- ComfyUI-H3-NVFP4-Fused-MLP (unless `fused_mlp` is disabled)
- ComfyUI-H3-Low-Memory-Sage2 for `exact_low_vram`
- ComfyUI-H3-CAB-Sampler
- ComfyUI-H3-Low-Step-Sigmas
- ComfyUI-KJNodes for `exact_speed`

Components are resolved at graph execution time, so custom-node import order is
irrelevant. Missing or outdated components produce an explicit error.
