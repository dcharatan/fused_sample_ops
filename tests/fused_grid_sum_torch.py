import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx, once_differentiable


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


class FusedGridSum(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        image: Float[Tensor, "batch channel height width"],
        samples: Float[Tensor, "batch sample_independent sample_summed 2"],
        weights: Float[Tensor, "batch head sample_independent sample_summed"],
    ) -> Float[Tensor, "batch head sample_independent channel"]:
        ctx.save_for_backward(image, samples, weights)

        grid_samples = F.grid_sample(
            image,
            samples,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return einsum(grid_samples, weights, "b c s s2, b hd s s2 -> b hd s c")

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        grad_output: Float[Tensor, "batch head sample_independent channel"],
    ) -> tuple[
        Float[Tensor, "batch channel height width"],
        Float[Tensor, "batch sample_independent sample_summed 2"],
        Float[Tensor, "batch head sample_independent sample_summed"],
    ]:
        image, samples, weights = ctx.saved_tensors

        import pydevd

        pydevd.settrace(suspend=False, trace_only_current_thread=True)

        print("in backward pass")

        return (
            torch.zeros_like(image),
            torch.zeros_like(samples),
            torch.zeros_like(weights),
        )


_fused_grid_sum_torch_manual = FusedGridSum.apply


def fused_grid_sum_torch_manual(
    image: Float[Tensor, "batch channel height width"],
    samples: Float[Tensor, "batch sample_independent sample_summed 2"],
    weights: Float[Tensor, "batch head sample_independent sample_summed"],
) -> Float[Tensor, "batch head sample_independent channel"]:
    return _fused_grid_sum_torch_manual(image, samples, weights)
