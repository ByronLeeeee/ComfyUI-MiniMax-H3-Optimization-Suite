from .nodes import H3CABSampler


NODE_CLASS_MAPPINGS = {"H3CABSampler": H3CABSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3CABSampler": "H3 CAB Low-Step Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
