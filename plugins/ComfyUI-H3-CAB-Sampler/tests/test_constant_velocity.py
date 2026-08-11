import importlib.util
import pathlib
import sys

import torch


COMFY_ROOT = pathlib.Path(
    r"C:\Users\demon\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI"
)
sys.path.insert(0, str(COMFY_ROOT))

from comfy.nested_tensor import NestedTensor  # noqa: E402


MODULE_PATH = pathlib.Path(__file__).parents[1] / "nodes.py"
SPEC = importlib.util.spec_from_file_location("h3_cab_nodes_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConstantVelocityModel:
    def __init__(self, velocity):
        self.velocity = velocity

    def __call__(self, sample, sigma, **kwargs):
        # Comfy samplers receive x0 (denoised), so x0 = x - sigma * dx/dsigma.
        return sample - self.velocity * sigma.reshape(-1, *([1] * 3))


def test_nested_constant_velocity_is_exact():
    start_tensor = torch.tensor([[[[2.0, -1.0], [0.5, 3.0]]]])
    velocity_tensor = torch.tensor([[[[0.25, -0.5], [1.0, 0.125]]]])
    start = NestedTensor([start_tensor])
    velocity = NestedTensor([velocity_tensor])
    sigmas = torch.tensor([1.0, 0.92, 0.80, 0.63, 0.0])

    result = MODULE.sample_h3_cab(
        ConstantVelocityModel(velocity),
        start,
        sigmas,
        disable=True,
        order=2,
        theta=0.2,
    )
    expected = start_tensor - velocity_tensor
    torch.testing.assert_close(result.tensors[0], expected, rtol=0, atol=2e-6)


if __name__ == "__main__":
    test_nested_constant_velocity_is_exact()
    print("CAB NestedTensor constant-velocity regression: PASS")
