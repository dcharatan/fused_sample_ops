import pytest
import torch
from jaxtyping import Float
from torch import Tensor


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0")


@pytest.fixture
def image_2x2(device) -> Float[Tensor, "batch channel height width"]:
    image = [
        [1, 10],
        [100, 1000],
    ]
    return torch.tensor(image, device=device, dtype=torch.float32)[None, None]


@pytest.fixture
def make_single_sample(device):
    def _make_single_sample(x: float | int, y: float | int):
        xy = torch.tensor((x, y), device=device, dtype=torch.float32)
        return xy[None, None, None]

    return _make_single_sample


@pytest.fixture
def single_weight(device):
    return torch.tensor(2, dtype=torch.float32, device=device)[None, None, None, None]
