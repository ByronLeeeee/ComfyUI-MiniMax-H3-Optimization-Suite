# ComfyUI H3 Long-Sequence VRAM Optimizer

Experimental MiniMax H3 model patch for long, high-resolution native ComfyUI
workflows. It does not reduce sampler steps or split the video timeline.

The node combines three peak-memory controls:

- token-chunked H3 MLP evaluation;
- runtime chunking of the dedicated MiniMax H3 Turbo LoRA bypass delta;
- an activation reserve hint for ComfyUI DynamicVRAM, so a long sequence keeps
  fewer weights resident and leaves more room for activations.

Short sequences stay on the original forward path. The default `auto` profile
detects total VRAM; `16gb` is the current exact validation target. These exact
profiles use LoRA chunking and DynamicVRAM activation reserve but leave the base
NVFP4 MLP calculation intact. Put the node after the Turbo LoRA and other H3
model patches, and before the guider/sampler.

`16gb_chunked` is an explicit last-resort profile. It also chunks the base MLP,
which greatly reduces the FC1 peak but makes NVFP4 derive a separate dynamic
input scale per chunk. It does not skip steps or split temporal attention, but
small numerical differences from the exact path are expected.

This trades some throughput for a lower peak. It is intended for jobs that
otherwise OOM; it is not expected to accelerate short clips.

Validated 16 GB fallback setting:

```text
profile: 16gb_chunked
mlp_chunk_rows: 4096
lora_chunk_mib: 256
manual_reserve_gib: 0
```

In a `1376x768 / 362-frame / 4-step` denoise-only stress test, the exact
`16gb` profile still OOMed while the setting above completed all four steps.
The MLP FC1 BF16/FP16 temporary upper bound fell from about 6106 MiB to 224
MiB per call. This test deliberately omitted VAE decode, and the chunked mode
is not bit-exact because NVFP4 dynamic input scales are derived per chunk.
