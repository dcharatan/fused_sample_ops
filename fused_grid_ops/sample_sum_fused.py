import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx, once_differentiable

from . import _cuda

TypeImages = Float[Tensor, "batch channel height width"]
TypeSamples = Float[Tensor, "batch query depth 2"]
TypeWeights = Float[Tensor, "batch head query depth"]
TypeOutputs = Float[Tensor, "batch head query channel"]


class SampleSumFused(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        images: TypeImages,
        samples: TypeSamples,
        weights: TypeWeights,
    ) -> TypeOutputs:
        # Save the inputs for the backward pass.
        ctx.save_for_backward(images, samples, weights)

        # Create an empty tensor for the outputs.
        b, c, _, _ = images.shape
        _, hd, q, _ = weights.shape
        outputs = torch.empty(
            (b, hd, q, c),
            dtype=images.dtype,
            device=images.device,
            requires_grad=images.requires_grad
            or samples.requires_grad
            or weights.requires_grad,
        )

        _cuda.sample_sum_forward(images, samples, weights, outputs)

        return outputs

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        output_gradients: TypeOutputs,
    ) -> tuple[TypeImages, None, TypeWeights]:
        # Retrieve the inputs to the forward pass.
        images, samples, weights = ctx.saved_tensors

        # Create empty tensors for the gradients.
        image_gradients = torch.zeros_like(images)
        sample_gradients = torch.zeros_like(samples)
        weight_gradients = torch.zeros_like(weights)

        _cuda.sample_sum_backward(
            output_gradients,
            images,
            samples,
            weights,
            image_gradients,
            sample_gradients,
            weight_gradients,
        )

        return image_gradients, sample_gradients, weight_gradients


_sample_sum_fused = SampleSumFused.apply


def sample_sum_fused(
    images: TypeImages,
    samples: TypeSamples,
    weights: TypeWeights,
) -> TypeOutputs:
    """Compute a fused combination of torch.nn.functional.grid_sample and summation
    across the depth dimension.
    """
    return _sample_sum_fused(images, samples, weights)
