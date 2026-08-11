# ComfyUI-H3-Step-Profiler

Temporary diagnostic node for profiling one native MiniMax H3 denoise call.
It records CUDA-event critical-path timings for attention, MLP, their linear
layers, and other work, plus a PyTorch CPU/CUDA trace containing copy and kernel
activity. Traces are written to `ComfyUI/output/profiles`.

The selected call is deliberately synchronized and will run slower. Do not use
the node for normal generation.

The package also contains a lightweight per-denoise CUDA timer, which records
absolute and incremental allocated/reserved CUDA peaks, plus two latent sinks.
The fingerprint sink avoids VAE decoding and writes exact SHA-256 hashes for
controlled numerical-equivalence benchmarks.
