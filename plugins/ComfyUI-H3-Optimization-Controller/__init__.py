from .nodes import H3OptimizationController, H3OptimizedSampling


NODE_CLASS_MAPPINGS = {
    "H3OptimizationController": H3OptimizationController,
    "H3OptimizedSampling": H3OptimizedSampling,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3OptimizationController": "H3 Optimization Controller",
    "H3OptimizedSampling": "H3 Optimized Sampling",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
