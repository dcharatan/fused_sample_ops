import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from .sample_dot_fused import (
    TypeDepths,
    TypeImages,
    TypeOutputs,
    TypeQueries,
    TypeSamples,
)


def sample_dot_torch(
    images: TypeImages,
    samples: TypeSamples,
    queries: TypeQueries,
    depths: TypeDepths,
    num_octaves: int,
) -> TypeOutputs:
    """The non-fused equivalent of sample_dot."""

    samples = F.grid_sample(
        images,
        samples,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    # Generate positional encoding frequencies and phases.
    frequencies = torch.arange(num_octaves, dtype=queries.dtype, device=queries.device)
    frequencies = 2 * torch.pi * 2**frequencies
    phases = torch.tensor([0, 1], dtype=queries.dtype, device=queries.device)
    phases = 0.5 * torch.pi * phases

    # Positionally encode the depths.
    _, _, d = depths.shape
    frequencies = repeat(frequencies, "f -> () (f p) () ()", p=2)
    phases = repeat(phases, "p -> () (f p) () ()", f=num_octaves)
    depths = rearrange(depths, "b q d -> b () q d")
    depths = torch.sin(depths * frequencies + phases)

    # Concatenate the positionally encoded depths onto the queries.
    queries = repeat(queries, "b q c -> b c q d", d=d)
    queries = torch.cat((queries, depths), dim=1)

    return einsum(samples, queries, "b c q d, b c q d -> b q d")
