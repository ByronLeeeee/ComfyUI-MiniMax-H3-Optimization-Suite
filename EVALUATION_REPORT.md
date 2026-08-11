# MiniMax H3 ComfyUI Optimization Node Evaluation

Evaluation date: 2026-08-12
Evaluation scope: one RTX 5070 Ti 16 GB workstation

## 1. Executive summary

The suite contains independent optimizations for different bottlenecks rather
than one universal acceleration switch:

1. **Exact full-step optimization.** At 1280×736, approximately 5 seconds, and
   20 steps, H3 NVFP4 Fused MLP reduced denoise time from 208.756 to 194.079
   seconds (-7.03%) and full prompt time from 244.969 to 223.674 seconds
   (-8.69%). Peak allocated VRAM fell by 712.246 MiB. The combined latent hash
   and decoded video were identical in this test.
2. **Attention-memory optimization.** On top of Fused MLP, Low-Memory Sage2
   reduced the measured peak from 4410.209 to 3843.612 MiB (-566.597 MiB), at a
   2.13% denoise-time cost relative to fused + stock Sage2.
3. **Low-step optimization.** Relative to the 20-step fused + Sage2 reference,
   CAB-2 at 12 steps reduced denoise time by 39.79%, with video SSIM 0.8153 and
   PSNR 20.24 dB. At 10 steps it reduced denoise time by 49.79%, with SSIM
   0.8060 and PSNR 19.02 dB.
4. **Long-sequence capacity optimization.** A 736×1280, 15-second, CAB-2
   14-step job completed all denoise steps and both VAE decodes on 16 GB using
   `Fused Kernels accurate + Low-Memory Sage2 + Long-Sequence 16gb_chunked`.
   The reported FC1 temporary upper bound changed from 5488.6 MiB to 224.0 MiB
   per chunk. This validates completion, not acceleration or 20-step quality.

For the official 20-step schedule, the two validated starting points are:

- speed: `exact_speed + res_multistep + stock_simple + 20 steps`;
- lower peak VRAM: `exact_low_vram + res_multistep + stock_simple + 20 steps`.

CAB saves model evaluations only when the step count is reduced. For a job that
already OOMs, try Long-Sequence `auto`, then `16gb`, and use `16gb_chunked` only
as a last resort. Base-MLP chunking changes the grouping used for NVFP4 dynamic
input scales and is not bit-exact to the unchunked path.

## 2. Test environment

| Item | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti 16 GB, Blackwell SM 12.0 |
| System memory | approximately 48 GiB; approximately 24 GB shared GPU memory available |
| Operating system | Windows |
| ComfyUI | 0.31.0 |
| Python | 3.12.9 |
| PyTorch | 2.10.0+cu130 |
| comfy-kitchen | 0.2.28 |
| Triton | 3.6.0 |
| UNet | `MiniMax_H3_FL2VA_pruned_nvfp4.safetensors` |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` |

Most controlled comparisons used text-to-video, 1280×736 or approximately
480p, and a 5-second duration. A separate 736×1280, 15-second job evaluated the
long-sequence capacity path.

## 3. Method

### 3.1 Timing

- Cold initialization, model loading, denoising, VAE decoding, and container
  encoding were kept conceptually separate.
- The progress-bar duration is used for denoise comparisons. Full prompt time
  also includes text encoding, dynamic model loading, decode, and muxing.
- Warm-up and dynamic-VRAM state can contaminate a single timing, so small
  differences should not be treated as conclusive without interleaved repeats.

### 3.2 Numerical identity

Independent SHA-256 fingerprints were generated for the audio and video latent
tensors and then combined. “Exact” means identical in the measured hardware,
software, and Sage2 path. It does not promise bit-exact output across GPUs,
PyTorch versions, or kernel implementations.

### 3.3 Low-step quality

The fixed-prompt, fixed-seed, fused + Sage2 `res_multistep` 20-step output was
used as the trajectory reference. Decoded frames were compared with SSIM and
PSNR. These metrics measure closeness to that reference, not aesthetics,
motion quality, anatomy, prompt adherence, editing, or audio quality.

## 4. Component results

### 4.1 H3 NVFP4 Fused MLP

Native H3 forms a large BF16/FP16 SwiGLU activation before quantizing the FC2
input to NVFP4. Fused MLP combines the activation and FC2-input quantization,
then continues to use comfy-kitchen's native NVFP4 GEMM. It skips no blocks and
does not reduce sampling steps.

Measured results:

- avoided an approximately 952.7 MiB BF16/FP16 intermediate per MLP call in the
  1280×736 workload;
- reduced 20-step denoise from 208.756 to 194.079 seconds (-7.03%);
- reduced full prompt time from 244.969 to 223.674 seconds (-8.69%);
- reduced peak allocated VRAM from 5122.455 to 4410.209 MiB;
- produced an identical combined latent hash and decoded video (SSIM 1,
  PSNR infinity).

Conclusion: this is the lowest-risk default optimization for the validated H3
NVFP4 checkpoint on Blackwell.

### 4.2 H3 Fused Kernels

This node targets H3 block modulation and gated residual work. The native path
launches operations separately for packed text, image, video, and audio
segments. The fused path maps every token to its modulation row and processes
the complete packed sequence. All 50 blocks, attention calls, MLP calls, and
sampling steps remain present.

- `auto + accurate` keeps PyTorch RMSNorm and the native multiply/add rounding
  boundary while fusing segment work.
- `fast` may additionally fuse RMSNorm + AdaLN and needs its own quality A/B.
- `auto` falls back to the PyTorch reference if Triton compilation or execution
  fails.
- This is not the NVFP4 Fused MLP node; the two optimize different boundaries.

The current runs did not establish a stable universal throughput improvement:
different shapes showed small gains or regressions. This report therefore
confirms compatibility with the Low-Memory Sage2 and Long-Sequence chain but
does not assign it a general speed percentage.

### 4.3 H3 Low-Memory Sage2

The node reorganizes Sage2 Q/K quantization and the PV path to reduce temporary
attention workspace. It does not change the number of model evaluations.

| Metric | Fused + Sage2 | Fused + Low-Memory Sage2 | Difference |
|---|---:|---:|---:|
| 20-step denoise | 194.079 s | 198.212 s | +4.133 s |
| Denoise peak allocated | 4410.209 MiB | 3843.612 MiB | -566.597 MiB |

The combined latent SHA-256 was identical. Low-Memory Sage2 was 2.13% slower
than fused + stock Sage2, but still 5.05% faster than the unfused Sage2
baseline. Its purpose is peak headroom, especially when resolution, duration,
or shared-memory paging approaches the card's limit.

### 4.4 H3 CAB low-step sampler

CAB-2/CAB-3 are training-free multistep solvers. They reuse velocity history
and apply defect correction without adding a model call inside a nominal step.
The implementation adapts the published equations to ComfyUI's denoised-output
convention and H3's packed audio/video latent.

Six-step comparisons against a same-seed 10-step reference:

| Resolution / duration | Six-step method | SSIM | PSNR |
|---|---|---:|---:|
| approximately 480p / 5 s | res_multistep | 0.7902 | 19.62 dB |
| approximately 480p / 5 s | CAB-2 | 0.8155 | 20.63 dB |
| 1280×736 / 5 s | res_multistep | 0.8017 | 20.09 dB |
| 1280×736 / 5 s | CAB-2 | 0.8270 | 21.26 dB |

Both methods used six model evaluations. CAB arithmetic was below 1% of
denoise time, while its history added approximately 112 MiB of peak memory at
736p. CAB-2 was slightly better than CAB-3 in this sample, and `theta=0.20` is
the validated default.

The 20-step-reference matrix at 1280×736 / 5 seconds:

| Method | Denoise | Full prompt | Peak MiB | SSIM / PSNR vs 20-step |
|---|---:|---:|---:|---:|
| res 20 reference | 194.079 s | 223.674 s | 4410.209 | 1.0000 / infinity |
| CAB-2 14 | 136.763 s | 137.504 s (decode omitted) | 4522.856 | 0.8274 / 21.66 dB |
| CAB-2 12 | 116.859 s | 146.589 s | 4522.856 | 0.8153 / 20.24 dB |
| CAB-2 10 | 97.438 s | 126.983 s | 4522.856 | 0.8060 / 19.02 dB |

The first CAB-14 run had an abnormal 61-second first-step load. The table uses
a warm no-decode repeat with the same latent hash.

![CAB low-step comparison at 2.5 seconds](benchmark_artifacts/media/cab_lowstep_frame_2p5s.png)

[Download the CAB 20/14/12/10-step 2x2 comparison video](benchmark_artifacts/media/cab_lowstep_2x2.mp4)

### 4.5 H3 Low-Step Sigmas

`stock_simple` reproduces ComfyUI's simple scheduler indices and the native H3
shift. The `balanced_late`, `strong_late`, and custom late-bias profiles did not
improve this sample. `stock_simple` remains the recommended production value.

### 4.6 Unified controller nodes

`H3 Optimization Controller` and `H3 Optimized Sampling` provide a compact
front end over the standalone plugins. The controller detects competing
attention overrides, and its `steps` output keeps the solver and sigma schedule
synchronized.

An end-to-end smoke test at approximately 0.2 MP, 1.6 seconds, and two CAB-2
steps succeeded for both `exact_low_vram` and `exact_speed`. Their combined
latent SHA-256 was the same. The observed 27.13-second cold run and 1.07-second
warm run are not a preset speed comparison because only the former contained
cold model initialization.

### 4.7 H3 Long-Sequence VRAM Optimizer and 15-second validation

This capacity guard combines:

- a DynamicVRAM activation-reserve hint;
- temporary chunking for the dedicated H3 Turbo-LoRA bypass delta, if present;
- optional token-row chunking of the base MLP in `16gb_chunked`.

Short sequences stay on the original forward path. `auto` and `16gb` leave the
base NVFP4 MLP unchunked and should be tried first. `16gb_chunked` changes
NVFP4 dynamic input scaling per chunk and is not bit-exact to the unchunked
path.

A similar 15-second chain without the Long-Sequence guard previously raised a
CUDA OOM during the first model forward's weight-prefetch stage, before CAB had
created its history. The following chain completed:

```text
H3 Fused Kernels: auto / accurate
-> H3 Low-Memory Sage2
-> H3 Long-Sequence: 16gb_chunked / 4096 rows / 256 MiB
-> CAB-2: theta 0.20 / simple / 14 steps
```

| Metric | Result |
|---|---:|
| Estimated / actual packed rows | 98,842 / 100,363 |
| Requested activation reserve | 5120 MiB |
| FC1 temporary upper bound | 5488.6 -> 224.0 MiB per chunk |
| 14-step denoise | approximately 940 s; 67.15 s/step average |
| Full prompt | 1064 s (00:17:44) |
| Output | 736×1280, 15.083 s, 24 fps, H.264 + stereo AAC |
| File | 5,252,611 bytes; SHA-256 `A6A042E0...A524FB` |

Sampled frames broadly followed the office-to-computer-to-police-car sequence,
and the protagonist remained recognizable. Screen and vehicle text was
unstable, while fine spatial and full temporal continuity were not measured.
There is no same-prompt, same-seed, same-shape 20-step reference, so SSIM/PSNR
is intentionally omitted and this run does not show 14-step equivalence.

![Contact sheet from the completed 15-second output](benchmark_artifacts/media/cab14_long_15s_contact.png)

The public 15-second workflow retains the validated node chain but uses a new
1280×736, fixed-seed English railway-platform prompt designed for global users.
It is a reproducible example, not a claim that the new prompt itself produced
the portrait benchmark above.

## 5. Applicability at the official 20 steps

| Configuration | Steps | Denoise | Full prompt | Peak MiB | SSIM / PSNR |
|---|---:|---:|---:|---:|---:|
| Stock KJ Sage2 | 20 | 208.756 s | 244.969 s | 5122.455 | 1.0000 / infinity |
| Fused + Sage2 | 20 | 194.079 s | 223.674 s | 4410.209 | 1.0000 / infinity |
| Fused + LowMem Sage2 | 20 | 198.212 s | 228.359 s | 3843.612 | 1.0000 / infinity |
| Fused + CAB-2 | 14 | 136.763 s | 137.504 s (decode omitted) | 4522.856 | 0.8274 / 21.66 dB |
| Fused + CAB-2 | 12 | 116.859 s | 146.589 s | 4522.856 | 0.8153 / 20.24 dB |
| Fused + CAB-2 | 10 | 97.438 s | 126.983 s | 4522.856 | 0.8060 / 19.02 dB |

| Component | Effect at 20 steps | Usage |
|---|---|---|
| NVFP4 Fused MLP | active inside every step | default on supported Blackwell/NVFP4 systems |
| H3 Fused Kernels | active inside every step; no stable universal speed result | use `auto + accurate` only in validated chains |
| Exact Sage2 | preserves the measured reference path | normal default |
| Low-Memory Sage2 | lowers peak workspace; saving is not multiplied by step count | use near the VRAM limit |
| Long-Sequence guard | reserves activation headroom; bypasses short sequences | use when long/high-resolution jobs OOM |
| CAB-2 | does not reduce NFE if left at 20 steps | use when reducing to 14/12/10 steps |
| Biased sigmas | no positive evidence in the current sample | keep `stock_simple` |

## 6. Validated configurations

### 6.1 Current KJ Sage2 quality control

```text
preset: exact_speed
fused_mlp: false
sampler: res_multistep
sigmas: stock_simple
steps: 20
```

### 6.2 Exact speed priority

```text
preset: exact_speed
sampler: res_multistep
sigmas: stock_simple
steps: 20
```

### 6.3 Conventional lower-VRAM path

```text
preset: exact_low_vram
sampler: res_multistep
sigmas: stock_simple
steps: 20
```

### 6.4 16 GB long-sequence OOM fallback

```text
H3 Fused Kernels: auto / accurate
H3 Low-Memory Sage2
H3 Long-Sequence: auto -> 16gb -> 16gb_chunked
16gb_chunked parameters: 4096 rows / 256 MiB / manual reserve 0
```

This configuration is not promised to be faster. Use `16gb_chunked` only when
the unchunked profiles cannot finish the job.

### 6.5 Low-step experiment

```text
preset: exact_speed
sampler: CAB-2
theta: 0.20
sigmas: stock_simple
steps: compare 14 -> 12 -> 10 against a 20-step reference
```

## 7. Scope and limitations

- Most quality comparisons use one prompt and one seed.
- SSIM/PSNR measures proximity to one reference trajectory, not aesthetic
  quality.
- The current set is not a large-sample statistical benchmark.
- It does not cover difficult dialogue, hand interaction, large casts, rapid
  editing, or strict audiovisual synchronization.
- Non-Blackwell hardware does not have a complete real-device validation
  matrix.
- The 15-second Long-Sequence test proves completion but has no controlled
  successful unchunked baseline or same-shape 20-step quality reference.
- `16gb_chunked` changes the NVFP4 dynamic-scaling groups and is approximate.
- H3 Fused Kernels does not yet have enough controlled repeat data for a
  deterministic speed claim.
- Dynamic weight loading and warm-up state affected individual timing runs;
  the report uses the same-hash warm CAB-14 repeat for its 5-second table.

## 8. External references

- CAB paper: <https://arxiv.org/abs/2605.16736>
- Official CAB implementation: <https://github.com/Anuska-Roy/CAB>
- SageAttention: <https://github.com/thu-ml/SageAttention>
- ComfyUI: <https://github.com/Comfy-Org/ComfyUI>
