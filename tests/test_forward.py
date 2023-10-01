import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from ..fused_grid_sum import fused_grid_sum_forward
from .fused_grid_sum_torch import fused_grid_sum_torch


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
    return torch.tensor(1, dtype=torch.float32, device=device)[None, None, None, None]


@pytest.fixture
def run_comparison(make_single_sample, image_2x2, single_weight):
    def _run_comparison(
        x: float | int,
        y: float | int,
    ) -> tuple[
        Float[Tensor, "batch head sample_independent channel"],
        Float[Tensor, "batch head sample_independent channel"],
    ]:
        sample = make_single_sample(x, y)
        expected = fused_grid_sum_forward(image_2x2, sample, single_weight)
        actual = fused_grid_sum_torch(image_2x2, sample, single_weight)
        assert torch.allclose(expected, actual, atol=5e-5)

    return _run_comparison


def test_interpolation_center(run_comparison):
    run_comparison(0, 0)


def test_interpolation_top_left(run_comparison):
    run_comparison(-0.25, -0.25)


def test_interpolation_top_right(run_comparison):
    run_comparison(0.25, -0.25)


def test_interpolation_bottom_left(run_comparison):
    run_comparison(-0.25, 0.25)


def test_interpolation_bottom_right(run_comparison):
    run_comparison(0.25, 0.25)


def test_interpolation_edge_left(run_comparison):
    run_comparison(-1, 0)


def test_interpolation_edge_right(run_comparison):
    run_comparison(1, 0)


def test_interpolation_edge_top(run_comparison):
    run_comparison(0, -1)


def test_interpolation_edge_bottom(run_comparison):
    run_comparison(0, 1)


def test_interpolation_far_left(run_comparison):
    run_comparison(-10, 0)


def test_interpolation_far_right(run_comparison):
    run_comparison(10, 0)


def test_interpolation_far_top(run_comparison):
    run_comparison(0, -10)


def test_interpolation_far_bottom(run_comparison):
    run_comparison(0, 10)


def test_interpolation_with_padding_top_left(run_comparison):
    run_comparison(-1.25, -1.25)


def test_interpolation_with_padding_top(run_comparison):
    run_comparison(0, -1.25)


def test_interpolation_with_padding_top_right(run_comparison):
    run_comparison(1.25, -1.25)


def test_interpolation_with_padding_left(run_comparison):
    run_comparison(-1.25, 0)


def test_interpolation_with_padding_right(run_comparison):
    run_comparison(1.25, 0)


def test_interpolation_with_padding_bottom_left(run_comparison):
    run_comparison(-1.25, 1.25)


def test_interpolation_with_padding_bottom(run_comparison):
    run_comparison(0, 1.25)


def test_interpolation_with_padding_bottom_right(run_comparison):
    run_comparison(1.25, 1.25)
