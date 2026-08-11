# ComfyUI-H3-NVFP4-Fused-MLP

Experimental MiniMax H3 optimization for native NVFP4 checkpoints on NVIDIA
Blackwell GPUs.

## What it changes

The stock H3 MLP performs:

1. NVFP4 FC1 GEMM, producing `[gate | up]` in BF16;
2. `silu(gate) * up`, materializing a large BF16 tensor;
3. a second read of that tensor to calculate NVFP4 scales and quantize it;
4. the native cuBLAS NVFP4 FC2 GEMM.

This node replaces steps 2-3 with Triton kernels that read the two FC1 halves
directly and emit FC2's packed E2M1 data plus UE4M3 block scales. FC2 itself is
still executed by `comfy_kitchen`'s native cuBLAS FP4 path. No transformer block
is skipped and the sampler/step count is unchanged.

The optimized quantizer processes eight 16-value blocks with two full warps per
program. It also folds tensor-scale rounding into the global reduction and only
zeros the small padded tail of the cuBLAS scale layout. These are scheduling and
memory-traffic changes only; the NVFP4 values are unchanged.

For the 480x864, 5-second H3 test shape, the packed sequence is roughly 16k
tokens. Avoiding the `[M, 14336]` BF16 activation saves about 435 MiB of transient
HBM traffic/storage per MLP invocation.

## Node

`sampling/minimax_h3 → H3 NVFP4 Fused MLP (Experimental)`

The released node exposes only the exact native-rounding path. It recalculates
the tensor-wide scale on every MLP call and matches the stock packed FP4 values
and block scales byte-for-byte. Earlier cached-scale experiments were removed
because small cross-step quantization changes caused visible H3 trajectory drift.

`vram_residency` controls an optional integration with ComfyUI's DynamicVRAM
weight loader:

- `off` (default): keep ComfyUI's original inference-memory estimate;
- `auto_safe`: return at most 50% of the conservatively calculated activation
  saving to the resident-weight budget, capped at 640 MiB;
- `auto_balanced`: return at most 72%, capped at 896 MiB.

The residency modes do not reserve memory, move tensors during a forward pass,
or alter the sampled latent. They only lower the memory estimate passed to the
loader before sampling, allowing it to retain more weights if that is useful.
The calculation uses a strict lower bound from H3's native packed latent shape
and retains safety headroom for CUDA workspaces. Loader estimates and tracked
resident weights are logged once for diagnosis.

Unsupported models, non-NVFP4 FC2 weights, dynamic FC2 LoRA patches, and runtime
kernel failures fall back to the stock MLP path and log the reason once.

## Compatibility

- Intended for RTX 50-series / SM120 and other Blackwell GPUs with working
  `comfy_kitchen` NVFP4 support.
- Requires PyTorch, Triton, and `comfy_kitchen` from the active ComfyUI runtime.
- It is a separate plugin and does not modify
  `ComfyUI-H3-Experimental-Optimizations` or `ComfyUI-H3-Fused-Kernels`.

## Status

Validated on an RTX 5070 Ti with the user's native H3 NVFP4 checkpoint:

- 480x864, 5 steps: decoded video and audio hashes exactly matched stock. Timing
  was inside normal run-to-run variance.
- 736x1280, 2 steps: decoded hashes exactly matched stock; sampling changed from
  24.16s stock to 22.08s fused in the controlled pair (about 8.6% faster), while
  avoiding a 948.7 MiB BF16 intermediate per MLP call.
- Optimized kernel microbenchmark (`16384 x 14336` activated shape): exact fused
  preprocessing measured 2.56ms versus 5.77ms for the stock activation and
  quantization path (2.26x). The previous fused kernel measured about 3.31ms.
- 736x1280, 10 steps, warm paired run: stock sampling was 100.9s (10.09s/step),
  optimized fused sampling was 98.9s (9.89s/step), about 2.0% faster. Decoded
  video and audio SHA256 hashes matched exactly.
- 736x1280, 5 seconds, 10 steps, Sage `auto`, RTX 5070 Ti residency test:
  `off` 98.1s, `auto_safe` 98.0s, and `auto_balanced` 98.2s sampling. Safe and
  balanced returned 467.3 MiB and 672.9 MiB respectively to the weight budget;
  measured peak GPU use was 15533/15552/15566 MiB. All three decoded video and
  audio SHA256 hashes matched exactly. There was no measurable speed benefit on
  this workload, so residency remains optional and defaults to `off`.

Triton compilation affects the first run only. Always compare warm, identical
seed/prompt/resolution/step runs when testing another machine or checkpoint.

The NVFP4 scale swizzle and E2M1 packing are adapted from comfy_kitchen's
Apache-2.0 Triton quantizer (Copyright NVIDIA Corporation and affiliates).
