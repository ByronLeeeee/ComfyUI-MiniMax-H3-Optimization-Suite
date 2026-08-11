from .nodes import H3LongSequenceLatentSink, H3LongSequenceVRAMOptimizer


NODE_CLASS_MAPPINGS = {
    "H3LongSequenceVRAMOptimizer": H3LongSequenceVRAMOptimizer,
    "H3LongSequenceLatentSink": H3LongSequenceLatentSink,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LongSequenceVRAMOptimizer": "H3 Long-Sequence VRAM Optimizer (Experimental)",
    "H3LongSequenceLatentSink": "H3 Long-Sequence Latent Sink (Testing)",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
