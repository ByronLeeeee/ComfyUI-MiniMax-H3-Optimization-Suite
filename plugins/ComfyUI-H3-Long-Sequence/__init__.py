from .nodes import H3LongSequenceVRAMOptimizer


NODE_CLASS_MAPPINGS = {
    "H3LongSequenceVRAMOptimizer": H3LongSequenceVRAMOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LongSequenceVRAMOptimizer": "H3 Long-Sequence VRAM Optimizer (Experimental)",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
