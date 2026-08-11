# ComfyUI-H3-Blackwell-Hybrid-Attention

Experimental MiniMax H3 attention node for NVIDIA Blackwell (SM 12.x).

`H3 Blackwell Hybrid Attention` can keep SageAttention2++ on the first and last
denoise calls while using the faster FP4 SageAttention3 kernel in the middle.
This follows the upstream SageAttention3 accuracy guidance for video diffusion:
protect precision-sensitive edge timesteps instead of applying FP4 attention
unconditionally.

Requirements:

- Native ComfyUI MiniMax H3 model
- NVIDIA Blackwell GPU
- `sageattention` 2.2
- `sageattn3` wheel matching Python, Torch, and CUDA

For Comfy's official Windows wheel index, select the version matching the
installed environment. Example for Python 3.12, Torch 2.10, CUDA 13.0:

```powershell
python -m pip install --no-deps --index-url https://comfy-org.github.io/wheels/ "sageattn3==1.0.0+cu130torch2.10"
```

For `res_multistep`, `expected_denoise_calls` normally equals the scheduler step
count. Other solvers may evaluate the model more than once per displayed step;
use the diagnostic timer to determine the correct count.

## RTX 5070 Ti validation

For native H3 at 1280x736, 5 seconds and 10 denoise calls, the stable per-call
time was about 9.80 seconds with Sage2++ and 9.05 seconds on Sage3 middle calls.
With one Sage2 edge call at each end, the projected/observed clean sampling
saving is about 6% while retaining full step count.

This path is approximate, not lossless. Against the same-seed Sage2 video, the
hybrid output measured about 0.794 SSIM / 20.03 dB PSNR. It remained visually
coherent in the tested action sequence, but composition changed. Use
`hybrid_sage2_edges`, `sage2_edge_calls=1`, and do an A/B on important prompts.
Use the separate Low-Memory Sage2 node instead when exact Sage2 output matters.
