# ComfyUI MiniMax H3 Optimization Suite

Experimental, modular optimizations for native MiniMax H3 audio/video generation
in ComfyUI. The suite focuses on three separate goals:

- faster full-step inference without a distilled LoRA;
- lower peak VRAM pressure at high resolution or long duration;
- improved low-step quality through a training-free solver.

The repository uses **one repository / multiple independent custom nodes**.
Nothing patches ComfyUI core files, and every optimization can be measured or
disabled separately.

> The measurements currently come from one RTX 5070 Ti 16 GB workstation and
> a limited prompt/seed set. Read the [Chinese evaluation report](EVALUATION_REPORT.zh-CN.md)
> before treating any result as universal.

## Unified nodes

Most users only need to place these two nodes in the graph:

1. **H3 Optimization Controller** — patches the model according to a preset and
   passes its step count downstream.
2. **H3 Optimized Sampling** — produces a synchronized `SAMPLER` and `SIGMAS`
   pair for `SamplerCustomAdvanced`.

```text
UNET Loader
  -> H3 Optimization Controller
       |-- model -> BasicGuider
       |-- model -> H3 Optimized Sampling
       `-- steps -> H3 Optimized Sampling
                       |-- sampler -> SamplerCustomAdvanced
                       `-- sigmas  -> SamplerCustomAdvanced
```

Controller presets:

| Preset | Composition | Numerical status |
|---|---|---|
| `exact_speed` | fused NVFP4 MLP + KJ Sage2 `auto` | exact against the tested Sage2 path |
| `exact_low_vram` | fused NVFP4 MLP + low-memory Sage2 | exact latent hash in validation |
| `off` | no model patch | control |

The validated low-step starting point is `CAB-2`, `theta=0.20`, and
`stock_simple` sigmas. For the official 20-step baseline, start with
`res_multistep` and `stock_simple`.

## 736p / 5-second / 20-step evaluation

Fixed prompt, seed, model and stock-simple sigmas on an RTX 5070 Ti 16 GB:

| Configuration | Denoise | Full prompt | Peak allocated VRAM | Output |
|---|---:|---:|---:|---|
| KJ Sage2 baseline | 208.756 s | 244.969 s | 5122.455 MiB | reference |
| Fused MLP + Sage2 | 194.079 s | 223.674 s | 4410.209 MiB | exact hash |
| Fused MLP + LowMem Sage2 | 198.212 s | 228.359 s | 3843.612 MiB | exact hash |

Fused MLP improved denoise time by 7.03% and reduced the measured peak by
712.246 MiB while keeping the latent and decoded video exact. Low-Memory Sage2
returned another 566.597 MiB at a 2.13% denoise-time cost relative to fused
stock Sage2.

- [CAB 20/14/12/10-step comparison video](benchmark_artifacts/media/cab_lowstep_2x2.mp4)
- [Raw profiler data and frame metrics](benchmark_artifacts/raw/performance_and_quality_results.csv)
- [Full Chinese evaluation report](EVALUATION_REPORT.zh-CN.md)

## Installation

This is a source monorepo. Copy the required directories from [`plugins/`](plugins)
into `ComfyUI/custom_nodes/`, keeping each plugin as its own directory, then
restart ComfyUI.

For all supported controller presets and sampling modes, copy the controller,
Fused MLP, Low-Memory Sage2, CAB Sampler, and Low-Step Sigmas directories.
Step Profiler is optional.
`exact_speed` additionally requires
[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes). Sage presets require
a SageAttention build compatible with the active Python, PyTorch, CUDA, and GPU.

Do not put another Sage/attention patch before **H3 Optimization Controller**.
The controller intentionally rejects competing overrides instead of silently
replacing them.

## Example workflow

Import
[`example_workflows/MiniMax_H3_Unified_Optimization_20step.json`](example_workflows/MiniMax_H3_Unified_Optimization_20step.json)
into ComfyUI. It is a 5-second text-to-video workflow configured for the
conservative 20-step starting point:

```text
exact_speed + res_multistep + stock_simple + 20 steps
```

The workflow retains the original model selectors, resolution selector, audio
and video VAE decode, and video save nodes. Change the controller preset instead
of adding another Sage Attention node in front of it.

## Hardware scope

| Component | Blackwell SM 12.x | Ada SM 8.9 | Other GPUs |
|---|---:|---:|---:|
| NVFP4 Fused MLP | yes | no | no |
| Low-Memory Sage2 | yes | yes | not validated |
| CAB sampler / sigma schedule | yes | yes | expected to be portable; not validated |
| Controller | yes | partial, with unsupported features disabled | depends on selected components |

## Included plugins

- `ComfyUI-H3-Optimization-Controller`
- `ComfyUI-H3-NVFP4-Fused-MLP`
- `ComfyUI-H3-Low-Memory-Sage2`
- `ComfyUI-H3-CAB-Sampler`
- `ComfyUI-H3-Low-Step-Sigmas`
- `ComfyUI-H3-Step-Profiler`

The older MLP-call skipping experiment is deliberately not included because it
uses a separate approximate strategy and remains under evaluation.

## References

- [CAB paper](https://arxiv.org/abs/2605.16736)
- [Official CAB reference implementation](https://github.com/Anuska-Roy/CAB)
- [SageAttention](https://github.com/thu-ml/SageAttention)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)

## License

The code in this repository is released under the [MIT License](LICENSE).
External packages and model files are not bundled and retain their own terms.
