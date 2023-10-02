import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx, once_differentiable

from . import _cuda

TypeImages = Float[Tensor, "batch channel height width"]
TypeSamples = Float[Tensor, "batch query depth 2"]
TypeQueries = Float[Tensor, "batch query channel-2*num_octaves"]
TypeDepths = Float[Tensor, "batch query depth"]
TypeResults = Float[Tensor, "batch query depth"]


class FusedGridSum(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        images: TypeImages,
        samples: TypeSamples,
        queries: TypeQueries,
        depths: TypeDepths,
        num_octaves: int,
    ) -> TypeResults:
        _, c_images, _, _ = images.shape
        _, _, c_queries = queries.shape
        assert c_images - c_queries == 2 * num_octaves

        # Save the inputs for the backward pass.
        ctx.save_for_backward(images, samples, queries)

        # Create an empty tensor for the result.
        b, q, d = depths.shape
        outputs = torch.empty(
            (b, q, d),
            dtype=images.dtype,
            device=images.device,
            requires_grad=images.requires_grad or queries.requires_grad,
        )

        _cuda.grid_sample_dot_forward(images, samples, queries, depths, outputs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        result_gradients: TypeResults,
    ) -> None:
        raise Exception("Not implemented!")


_grid_sample_dot = FusedGridSum.apply


def grid_sample_dot(
    images: TypeImages,
    samples: TypeSamples,
    queries: TypeQueries,
    depths: TypeDepths,
    num_octaves: int,
) -> TypeResults:
    """Compute a fused combination of torch.nn.functional.grid_sample and dot product.
    This function only supports gradients for images and queries (not samples).
    """
    return _grid_sample_dot(images, samples, queries, depths, num_octaves)
