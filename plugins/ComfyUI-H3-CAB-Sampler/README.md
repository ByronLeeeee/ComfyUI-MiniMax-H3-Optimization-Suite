# ComfyUI-H3-CAB-Sampler

An experimental ComfyUI implementation of Corrected Adams-Bashforth (CAB-2
and CAB-3), adapted to native MiniMax H3's packed audio/video `NestedTensor`.

CAB is a training-free low-NFE solver published in May 2026. It reuses prior
velocity evaluations, adds a defect correction, and therefore does not add
model calls. The official paper reports improved quality/NFE trade-offs at
6-20 evaluations and includes flow models and HunyuanVideo evaluation.

- Paper: <https://arxiv.org/abs/2605.16736>
- Official reference implementation: <https://github.com/Anuska-Roy/CAB>

## Suggested starting point

- `CAB-2`
- `theta = 0.20`
- 6-10 steps
- stock `simple` H3 sigmas

Connect the sampler output to `SamplerCustomAdvanced`. CAB retains several
small latent-history buffers, so its sampler-state VRAM is slightly higher than
`res_multistep`, while the H3 model/attention peak remains dominant.

This implementation is independent of the H3 MLP, attention,
Low-Memory Sage2, and sigma-schedule plugins. It ports the published equations
to Comfy's denoised-output convention and H3's nested audio/video latent type;
it does not copy or modify ComfyUI core files.

## RTX 5070 Ti validation

Same prompt, seed and stock H3 simple sigmas, using exact Sage2 attention:

- 480p / 5 seconds: six-step `res_multistep` reached 0.7902 SSIM / 19.62 dB
  against the ten-step reference; six-step CAB-2 reached 0.8155 SSIM / 20.63 dB.
- 1280x736 / 5 seconds: six-step `res_multistep` reached 0.8017 SSIM / 20.09 dB;
  six-step CAB-2 reached 0.8270 SSIM / 21.26 dB.
- Both use six model calls, so sampling is roughly 40% shorter than ten steps.
  CAB's arithmetic overhead was below 1% of denoise time.
- CAB-2 was slightly better than CAB-3 on this sequence. `theta=0.20` is the
  validated default.
- The optimized CAB history peaked about 112 MiB above `res_multistep` at
  736p. Combined with Low-Memory Sage2, total measured denoise allocation peak
  remained about 454 MiB below stock Sage2.

These are single-prompt/single-seed trajectory comparisons, not a universal
quality score. Six-step CAB still differs from ten steps; use eight or ten
steps for difficult prompts and compare motion/audio before adopting it.
