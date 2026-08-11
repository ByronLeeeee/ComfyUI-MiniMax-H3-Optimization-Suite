# ComfyUI-H3-Low-Memory-Sage2

Experimental native MiniMax H3 attention node for reducing transient CUDA
memory without changing SageAttention2's quantization or attention kernels.

## What it changes

The native H3 projection produces Q, K, and V in one large BF16 allocation.
Stock Sage2 keeps that allocation alive while creating quantized Q/K, a
transposed V scratch buffer, quantized V, and the BF16 attention output. This
node performs the same RMSNorm, RoPE, smooth-K, Q/K INT8 quantization, V FP8
quantization, and Sage2 kernel, but after Q/K quantization it copies only V to
a compact buffer and releases the three-part QKV allocation early.

At 1280x736 / 5 seconds, the full H3 QKV projection is about 1.40 GiB and the
compact V is about 0.47 GiB. The expected reduction in the attention-local peak
is roughly 0.7 GiB. The extra device-to-device V copy occurs once per H3 block,
so the node primarily targets VRAM headroom rather than speed.

## Usage

Connect it after the model loader and optional H3 NVFP4 Fused MLP node, then
connect its output directly to the guider/sampler. Remove or bypass `Patch Sage
Attention KJ`, Blackwell Hybrid Attention, and other attention replacements.

Supported GPU paths match Sage2's FP8 CUDA implementation: Ada SM 8.9 and
Blackwell SM 10.0/12.x. The installed SageAttention wheel must match ComfyUI's
Torch and CUDA versions.

This package is independent from the existing H3 MLP and Hybrid Attention
plugins and does not overwrite either one.

## RTX 5070 Ti validation

Controlled 1280x736 / 5-second / two-step runs, same seed and warm model:

- stock Sage2 peak allocated memory: 4409.0 MiB;
- this node: 3842.4 MiB;
- reduction: 566.6 MiB (12.9% of the measured denoise allocation peak);
- all five baseline/repeat/fused final latent SHA-256 hashes were identical;
- steady denoise time changed from about 9.81 to 9.91 seconds per call (about
  1% slower).

With the CAB low-step sampler's history buffers active, the optimized peak was
3955.1 MiB, still about 454 MiB below stock Sage2. This is a headroom feature:
use it when resolution/duration approaches the VRAM limit; prefer ordinary
Sage2 if the extra half-GiB is unnecessary and every last percent of speed is
more valuable.
