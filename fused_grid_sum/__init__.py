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
        result = torch.empty((b, hd, s, c), dtype=images.dtype, device=images.device)

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

        _cuda.backward(
            result_gradients,
            images,
            image_gradients,
            samples,
            weights,
            weight_gradients,
        )

        return image_gradients, None, weight_gradients
