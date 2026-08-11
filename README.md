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
| `balanced_fast` | fused MLP + Sage2 edge calls + Sage3 middle calls | approximate |
| `maximum_speed` | fused MLP + Sage3 for every call | approximate |
| `off` | no model patch | control |

The validated low-step starting point is `CAB-2`, `theta=0.20`, and
`stock_simple` sigmas. For the official 20-step baseline, start with
`res_multistep` and `stock_simple`.

## Installation

This is a source monorepo. Copy the required directories from [`plugins/`](plugins)
into `ComfyUI/custom_nodes/`, keeping each plugin as its own directory, then
restart ComfyUI.

For all controller presets and sampling modes, copy all seven directories.
`exact_speed` additionally requires
[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes). Sage presets require
a SageAttention build compatible with the active Python, PyTorch, CUDA, and GPU.

Do not put another Sage/attention patch before **H3 Optimization Controller**.
The controller intentionally rejects competing overrides instead of silently
replacing them.

## Hardware scope

| Component | Blackwell SM 12.x | Ada SM 8.9 | Other GPUs |
|---|---:|---:|---:|
| NVFP4 Fused MLP | yes | no | no |
| Hybrid / Sage3 attention | yes | no | no |
| Low-Memory Sage2 | yes | yes | not validated |
| CAB sampler / sigma schedule | yes | yes | expected to be portable; not validated |
| Controller | yes | partial, with unsupported features disabled | depends on selected components |

## Included plugins

- `ComfyUI-H3-Optimization-Controller`
- `ComfyUI-H3-NVFP4-Fused-MLP`
- `ComfyUI-H3-Blackwell-Hybrid-Attention`
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

