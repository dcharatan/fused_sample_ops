import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx, once_differentiable

from . import _cuda

TypeImages = Float[Tensor, "batch channel height width"]
TypeSamples = Float[Tensor, "batch sample_independent sample_summed 2"]
TypeWeights = Float[Tensor, "batch head sample_independent sample_summed"]
TypeResults = Float[Tensor, "batch head sample_independent channel"]


class FusedGridSum(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        images: TypeImages,
        samples: TypeSamples,
        weights: TypeWeights,
    ) -> TypeResults:
        # Save the inputs for the backward pass.
        ctx.save_for_backward(images, samples, weights)

        # Create an empty tensor for the result.
        b, c, _, _ = images.shape
        _, hd, s, _ = weights.shape
        result = torch.empty(
            (b, hd, s, c),
            dtype=images.dtype,
            device=images.device,
            requires_grad=images.requires_grad or weights.requires_grad,
        )

        _cuda.forward(result, images, samples, weights)

        return result

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        result_gradients: TypeResults,
    ) -> tuple[TypeImages, None, TypeWeights]:
        # Retrieve the inputs to the forward pass.
        images, samples, weights = ctx.saved_tensors

        # Create empty tensors for the gradients. Note that we don't return a gradient
        # for the sample X/Y locations, since we don't need it.
        image_gradients = torch.zeros_like(images)
        weight_gradients = torch.zeros_like(weights)

        # We make the result gradients contiguous so that we don't have to deal with
        # icky broadcasting problems (strides that are zero).
        _cuda.backward(
            result_gradients.contiguous(),
            images,
            image_gradients,
            samples,
            weights,
            weight_gradients,
        )

        return image_gradients, None, weight_gradients


_fused_grid_sum = FusedGridSum.apply


def fused_grid_sum(
    images: TypeImages,
    samples: TypeSamples,
    weights: TypeWeights,
) -> TypeResults:
    """Compute a fused combination of torch.nn.functional.grid_sample and summation
    across the sample_summed dimension.
    """
    return _fused_grid_sum(images, samples, weights)
