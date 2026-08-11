# ComfyUI-H3-Low-Step-Sigmas

Experimental sigma-schedule laboratory for native MiniMax H3 when using 5-10
denoise evaluations without a distilled/4-step LoRA.

H3's video flow shift is 12. With very few stock `simple` steps, almost every
evaluation therefore remains at a high shifted sigma, followed by one large
jump to zero. This node power-warps the unshifted base-time grid before applying
the model's own shift. A bias above 1 retains the noisy start but spends more of
the limited budget on later, cleaner denoising.

Profiles:

- `stock_simple` (1.00): exact simple-scheduler grid, useful as a control.
- `balanced_late` (1.25): modest late-denoise reallocation.
- `strong_late` (1.50): stronger detail bias; motion may change more.
- `custom`: uses the `late_bias` value.

Connect the `SIGMAS` output to `SamplerCustomAdvanced`. Start with 6-8 steps and
`res_multistep`. The schedule does not reduce the number of model calls by
itself and is independent of the MLP, attention, and Low-Memory Sage2
plugins.

## Validation note

On the included 480p / 5-second H3 test, six-step `stock_simple` was closer to a
ten-step reference (SSIM 0.790) than `balanced_late` (0.770) or `strong_late`
(0.728). Therefore `stock_simple` is the safe default; the biased profiles are
kept only for prompt-specific experiments and are not claimed as optimizations.
For the validated low-step quality improvement, use the separate H3 CAB sampler
with stock sigmas.
