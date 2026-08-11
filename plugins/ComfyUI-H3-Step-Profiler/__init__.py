from .nodes import (
    H3BenchmarkLatentSink,
    H3DenoiseTimer,
    H3LatentFingerprintSink,
    H3StepProfiler,
)


NODE_CLASS_MAPPINGS = {
    "H3StepProfiler": H3StepProfiler,
    "H3DenoiseTimer": H3DenoiseTimer,
    "H3BenchmarkLatentSink": H3BenchmarkLatentSink,
    "H3LatentFingerprintSink": H3LatentFingerprintSink,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3StepProfiler": "H3 Single-Step Profiler (Diagnostic)",
    "H3DenoiseTimer": "H3 Denoise Timer (Diagnostic)",
    "H3BenchmarkLatentSink": "H3 Latent Sink (Benchmark)",
    "H3LatentFingerprintSink": "H3 Latent Fingerprint (Benchmark)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
