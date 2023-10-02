import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from ..fused_grid_sum import fused_grid_sum
from .common import *  # noqa F403
from .fused_grid_sum_torch import fused_grid_sum_torch


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

        image_expected = image_2x2.clone().requires_grad_(True)
        expected = fused_grid_sum(image_expected, sample, single_weight)
        expected.sum().backward()

        image_actual = image_2x2.clone().requires_grad_(True)
        actual = fused_grid_sum_torch(image_actual, sample, single_weight)
        actual.sum().backward()

        assert torch.allclose(image_expected.grad, image_actual.grad, atol=5e-5)

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
