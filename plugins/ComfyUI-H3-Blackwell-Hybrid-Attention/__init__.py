from .nodes import H3BlackwellHybridAttention


NODE_CLASS_MAPPINGS = {
    "H3BlackwellHybridAttention": H3BlackwellHybridAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3BlackwellHybridAttention": "H3 Blackwell Hybrid Attention",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
