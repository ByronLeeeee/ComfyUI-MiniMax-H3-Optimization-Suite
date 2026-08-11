# ComfyUI MiniMax H3 Optimization Suite

Modular optimization nodes for ComfyUI's native MiniMax H3 audio/video model.
The suite targets three different problems without modifying ComfyUI core:

- faster full-step NVFP4 inference without reducing model evaluations;
- lower peak VRAM and successful long/high-resolution generation;
- better quality per model evaluation when deliberately reducing the step count.

Each component is an independent custom node, so it can be enabled, bypassed,
and measured separately. Results and limitations are documented in the
[English evaluation report](EVALUATION_REPORT.md) and
[Chinese evaluation report](EVALUATION_REPORT.zh-CN.md).

> Validation currently comes from one RTX 5070 Ti 16 GB workstation and a
> limited prompt/seed set. The numbers are evidence for those workloads, not a
> universal performance guarantee.

## Which node should I use?

| Goal | Recommended starting point | Important trade-off |
|---|---|---|
| Official-style 20 steps, fastest validated exact path | `H3 NVFP4 Fused MLP` + KJ Sage2 | Blackwell/NVFP4 only |
| Official-style 20 steps, lower peak VRAM | `H3 NVFP4 Fused MLP` + `H3 Low-Memory Sage2` | about 2% slower than the fused + stock-Sage2 path in the measured run |
| 15-second/high-resolution job that otherwise OOMs on 16 GB | `H3 Fused Kernels` + `H3 Low-Memory Sage2` + `H3 Long-Sequence VRAM Optimizer` | `16gb_chunked` is a capacity fallback, not a speed node, and is not bit-exact |
| Fewer model evaluations | `H3 CAB Sampler`, starting with CAB-2, theta 0.20, 14 steps | less VRAM headroom and increasing deviation from the 20-step trajectory |
| Diagnose a workflow | `H3 Step Profiler` | measurement overhead; do not leave it enabled for production timing |

The official 20-step workflow still benefits from Fused MLP and Low-Memory
Sage2 because both operate inside every model evaluation. CAB only saves time
when the number of steps is actually reduced.

## Node reference

### H3 NVFP4 Fused MLP

Fuses the SwiGLU activation and FC2-input NVFP4 quantization in MiniMax H3's
MLP. It avoids constructing the largest BF16/FP16 intermediate, continues to
use comfy-kitchen's native NVFP4 GEMM, and does not skip blocks or steps.

- Best use: the default optimization on a Blackwell GPU with the supported H3
  NVFP4 checkpoint.
- Validated effect at 1280×736, 5 seconds, 20 steps: denoise `208.756 s ->
  194.079 s`, peak allocated VRAM `5122.455 -> 4410.209 MiB`.
- Numerical result in that validation: identical latent hash and decoded video.
- Scope: this is the Blackwell/NVFP4-specific fused node.

### H3 Fused Kernels

Replaces per-segment H3 AdaLN modulation and gated-residual launches with
whole-packed-sequence kernels. It keeps all 50 blocks, attention calls, MLP
calls, and sampling steps.

- `auto + accurate` is the recommended mode. It keeps the native RMSNorm and
  the multiply/add rounding boundary while fusing segment work.
- `fast` additionally attempts RMSNorm + AdaLN fusion and needs a separate
  quality check.
- Triton is optional for plugin loading; unsupported paths fall back to the
  PyTorch reference implementation in `auto` mode.
- This is different from **H3 NVFP4 Fused MLP**: it optimizes block modulation
  and gating rather than the NVFP4 MLP activation/quantization boundary.
- Its throughput effect depends on sequence shape and launch overhead. The
  current report does not claim a universal speed-up.

### H3 Low-Memory Sage2

Reorganizes Sage2 Q/K quantization and the PV path to reduce attention
workspace. It does not reduce steps.

- Validated effect on top of Fused MLP: peak `4410.209 -> 3843.612 MiB`, saving
  `566.597 MiB`.
- Cost in that run: denoise `194.079 -> 198.212 s` (+2.13%).
- Numerical result in the validated path: identical combined latent hash.
- Best use: high resolution, long clips, or any workflow close to the dedicated
  VRAM limit.

### H3 Long-Sequence VRAM Optimizer

A capacity guard for unusually long or high-resolution H3 sequences. It
combines a DynamicVRAM activation-reserve hint, chunking of the dedicated H3
Turbo-LoRA bypass delta when present, and an optional chunked base MLP path.
Short sequences automatically keep the original forward path.

- `auto` / `16gb`: leave the base NVFP4 MLP unchunked and are the conservative
  profiles to try first.
- `16gb_chunked`: last-resort 16 GB fallback. With 4096 rows in the validated
  15-second run, the per-call FC1 temporary upper bound fell from about
  `5488.6 MiB` to `224.0 MiB`.
- It does not split the video timeline and does not reduce sampling steps.
- Base-MLP chunking derives NVFP4 dynamic input scales per chunk, so
  `16gb_chunked` is not bit-exact to the unchunked path.
- It can be slower. Its purpose is to finish a job that otherwise runs out of
  memory.

### H3 CAB Sampler

A training-free corrected Adams-Bashforth solver adapted to ComfyUI's
denoised-output convention and H3's packed audio/video latent. It reuses recent
velocity history and performs defect correction without adding another model
call inside a nominal step.

- Validated starting point: `CAB-2`, `theta=0.20`, `stock_simple` sigmas.
- At 1280×736 / 5 seconds, CAB-2 at 14, 12, and 10 steps reduced model
  evaluations by 30%, 40%, and 50% relative to 20 steps.
- CAB history added about `112 MiB` peak memory at 736p in the measured run.
- CAB improves quality per evaluation relative to same-step `res_multistep` in
  the tested six-step cases; it does not make a low-step result identical to
  the official 20-step trajectory.

### H3 Low-Step Sigmas

Produces H3 sigma schedules. `stock_simple` reproduces ComfyUI's simple
scheduler and is the production default. The late-biased schedules remain
research controls and did not outperform `stock_simple` in the current sample.

### H3 Optimization Controller and H3 Optimized Sampling

A compact front end for the validated 5-second paths. The controller applies
the selected Fused MLP and Sage2 implementation; the sampling node emits a
synchronized `SAMPLER` and `SIGMAS` pair so the solver and schedule cannot use
different step counts.

Presets:

| Preset | Composition |
|---|---|
| `exact_speed` | NVFP4 Fused MLP + KJ Sage2 `auto` |
| `exact_low_vram` | NVFP4 Fused MLP + Low-Memory Sage2 |
| `off` | no model patch; passes the step count through |

Do not place another attention patch before the controller. It rejects
competing overrides instead of silently replacing them.

### H3 Step Profiler

Records step timing, CUDA peak allocation, and output fingerprints for
controlled A/B tests. Profiling synchronizes CUDA and therefore changes timing;
it is a diagnostics node, not an inference optimization.

## Measured results

### 1280×736, 5 seconds, 20 steps

Fixed prompt, seed, model, and stock-simple schedule on RTX 5070 Ti 16 GB:

| Configuration | Denoise | Full prompt | Peak allocated VRAM | Output |
|---|---:|---:|---:|---|
| KJ Sage2 baseline | 208.756 s | 244.969 s | 5122.455 MiB | reference |
| Fused MLP + Sage2 | 194.079 s | 223.674 s | 4410.209 MiB | exact hash |
| Fused MLP + Low-Memory Sage2 | 198.212 s | 228.359 s | 3843.612 MiB | exact hash |

### 736×1280, 15 seconds, CAB-14 capacity validation

The chain `Fused Kernels accurate -> Low-Memory Sage2 -> Long-Sequence
16gb_chunked -> CAB-2` completed all 14 denoise steps plus both audio and video
VAE decodes on the same 16 GB GPU. Denoise took about 940 seconds and the full
prompt took 1064 seconds. This validates completion, not speed or equivalence
to a same-shape 20-step reference.

![Sampled frames from the completed 15-second run](benchmark_artifacts/media/cab14_long_15s_contact.png)

## Installation

Copy each required directory from [`plugins/`](plugins) into
`ComfyUI/custom_nodes/`, keeping every plugin as a separate directory, then
restart ComfyUI.

For the compact 20-step workflow, install:

- `ComfyUI-H3-Optimization-Controller`
- `ComfyUI-H3-NVFP4-Fused-MLP`
- `ComfyUI-H3-Low-Memory-Sage2`
- `ComfyUI-H3-CAB-Sampler`
- `ComfyUI-H3-Low-Step-Sigmas`
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) for `exact_speed`

For the 15-second capacity example, also install:

- `ComfyUI-H3-Fused-Kernels`
- `ComfyUI-H3-Long-Sequence`

Sage presets require a SageAttention build compatible with the active Python,
PyTorch, CUDA, and GPU. Model weights and external dependencies are not bundled.

## Example workflows

### Conservative 20-step workflow

[`MiniMax_H3_Unified_Optimization_20step.json`](example_workflows/MiniMax_H3_Unified_Optimization_20step.json)
is a 5-second text-to-video workflow configured as:

```text
exact_speed + res_multistep + stock_simple + 20 steps
```

### 15-second global continuity and capacity test

[`MiniMax_H3_CAB14_LongSequence_15s.json`](example_workflows/MiniMax_H3_CAB14_LongSequence_15s.json)
uses the official single-node MiniMax H3 layout, 1280×736, 15 seconds, a fixed
seed, CAB-2 at 14 steps, and the 16 GB long-sequence fallback. Its English
prompt follows a traveler, a dog, and a red ticket across four shots and tests:

- identity, wardrobe, prop, and spatial continuity;
- low tracking and arc camera motion;
- readable text (`"PLATFORM 4"`) and a short English line;
- synchronized ambience, action sound, dialogue, train audio, and score.

The workflow is intentionally demanding. `16gb_chunked` favors successful
completion over speed and bit-exactness. Users with more VRAM should try the
`auto` or `16gb` profile before the chunked fallback.

## Hardware scope

| Component | Blackwell SM 12.x | Ada SM 8.9 | Other GPUs |
|---|---:|---:|---:|
| NVFP4 Fused MLP | validated | unsupported | unsupported |
| H3 Fused Kernels | validated | expected with compatible Triton; not validated | PyTorch fallback or compatible Triton; not validated |
| Low-Memory Sage2 | validated | design-compatible; not fully benchmarked | requires a compatible SageAttention path |
| Long-Sequence VRAM Optimizer | validated | expected; not validated | expected; not validated |
| CAB sampler / sigma schedule | validated | expected portable | expected portable |
| Controller | validated | partial; unsupported components are disabled | depends on selected components |

## Included plugins

- `ComfyUI-H3-Optimization-Controller`
- `ComfyUI-H3-NVFP4-Fused-MLP`
- `ComfyUI-H3-Fused-Kernels`
- `ComfyUI-H3-Low-Memory-Sage2`
- `ComfyUI-H3-Long-Sequence`
- `ComfyUI-H3-CAB-Sampler`
- `ComfyUI-H3-Low-Step-Sigmas`
- `ComfyUI-H3-Step-Profiler`

The earlier MLP-call skipping experiment and the rejected Turbo/4-step example
are not distributed in this repository.

## References

- [CAB paper](https://arxiv.org/abs/2605.16736)
- [Official CAB reference implementation](https://github.com/Anuska-Roy/CAB)
- [SageAttention](https://github.com/thu-ml/SageAttention)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)

## License

The code in this repository is released under the [MIT License](LICENSE).
External packages and model files are not bundled and retain their own terms.
