from .nodes import H3FusedKernels


NODE_CLASS_MAPPINGS = {
    "H3FusedKernels": H3FusedKernels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FusedKernels": "H3 Fused Kernels (Experimental)",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
