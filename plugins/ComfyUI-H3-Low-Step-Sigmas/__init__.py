from .nodes import H3LowStepSigmas


NODE_CLASS_MAPPINGS = {"H3LowStepSigmas": H3LowStepSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LowStepSigmas": "H3 Low-Step Sigma Schedule",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
