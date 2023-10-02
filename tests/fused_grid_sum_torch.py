import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def fused_grid_sum_torch(
    image: Float[Tensor, "batch channel height width"],
    samples: Float[Tensor, "batch sample_independent sample_summed 2"],
    weights: Float[Tensor, "batch head sample_independent sample_summed"],
) -> Float[Tensor, "batch head sample_independent channel"]:
    grid_samples = F.grid_sample(
        image,
        samples,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return einsum(grid_samples, weights, "b c s s2, b hd s s2 -> b hd s c")
