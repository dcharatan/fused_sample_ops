import torch
from jaxtyping import Float
from torch import Tensor

from . import _cuda


def fused_grid_sum_forward(
    image: Float[Tensor, "batch channel height width"],
    samples: Float[Tensor, "batch sample_independent sample_summed 2"],
    weights: Float[Tensor, "batch sample_independent head sample_summed"],
) -> Float[Tensor, "batch sample_independent head channel"]:
    b, c, _, _ = image.shape
    _, s, hd, _ = weights.shape
    result = torch.empty((b, s, c, hd), dtype=image.dtype, device=image.device)
    return _cuda.forward(result, image, samples, weights)
