import torch
from jaxtyping import Float
from torch import Tensor

from . import _cuda


def fused_grid_sum_forward(
    image: Float[Tensor, "batch channel height width"],
    samples: Float[Tensor, "batch sample_independent sample_summed 2"],
    weights: Float[Tensor, "batch head sample_independent sample_summed"],
) -> Float[Tensor, "batch head sample_independent channel"]:
    b, c, _, _ = image.shape
    _, hd, s, _ = weights.shape
    result = torch.zeros((b, hd, s, c), dtype=image.dtype, device=image.device)
    _cuda.forward(result, image, samples, weights)
    return result
