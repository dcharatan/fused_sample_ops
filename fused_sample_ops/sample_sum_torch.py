import torch.nn.functional as F
from einops import einsum

from .sample_sum_fused import TypeImages, TypeOutputs, TypeSamples, TypeWeights


def sample_sum_torch(
    images: TypeImages,
    samples: TypeSamples,
    weights: TypeWeights,
) -> TypeOutputs:
    grid_samples = F.grid_sample(
        images,
        samples,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return einsum(grid_samples, weights, "b c q d, b hd q d -> b hd q c")
