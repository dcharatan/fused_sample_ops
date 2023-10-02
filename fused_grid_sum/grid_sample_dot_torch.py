import torch.nn.functional as F
from einops import einsum, rearrange

from .grid_sample_dot import TypeImages, TypeQueries, TypeResults, TypeSamples


def grid_sample_dot_torch(
    images: TypeImages,
    samples: TypeSamples,
    queries: TypeQueries,
) -> TypeResults:
    """The non-fused equivalent of grid_sample_dot."""

    samples = F.grid_sample(
        images,
        rearrange(samples, "b s xy -> b s () xy"),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    samples = rearrange(samples, "b c s () -> b s c")
    return einsum(samples, queries, "b s c, b s c -> b s")
