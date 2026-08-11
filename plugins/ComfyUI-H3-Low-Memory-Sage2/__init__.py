from .nodes import H3LowMemorySage2Attention


NODE_CLASS_MAPPINGS = {
    "H3LowMemorySage2Attention": H3LowMemorySage2Attention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LowMemorySage2Attention": "H3 Low-Memory Sage2 Attention",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
