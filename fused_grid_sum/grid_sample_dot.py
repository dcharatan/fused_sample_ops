import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx, once_differentiable

from . import _cuda

TypeImages = Float[Tensor, "batch channel height width"]
TypeSamples = Float[Tensor, "batch sample_independent sample_summed 2"]
TypeResults = Float[Tensor, "batch sample_independent sample_summed channel"]


class FusedGridSum(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        images: TypeImages,
        samples: TypeSamples,
    ) -> TypeResults:
        # Save the inputs for the backward pass.
        ctx.save_for_backward(images, samples)

        # Create an empty tensor for the result.
        b, c, _, _ = images.shape
        _, h, w, _ = samples.shape
        outputs = torch.empty(
            (b, c, h, w),
            dtype=images.dtype,
            device=images.device,
            requires_grad=images.requires_grad or samples.requires_grad,
        )

        _cuda.grid_sample_dot_forward(images, samples, outputs)

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
) -> TypeResults:
    """Compute a fused combination of torch.nn.functional.grid_sample and dot product."""
    return _grid_sample_dot(images, samples)
