from time import time

import torch
from jaxtyping import install_import_hook
from tqdm import trange

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from fused_grid_sum import fused_grid_sum, grid_sample_dot
    from tests.fused_grid_sum_torch import fused_grid_sum_torch


if __name__ == "__main__":
    device = torch.device("cuda:0")

    b = 8
    c = 128
    h = 129
    w = 130
    s = 131
    s2 = 132
    hd = 8

    def rand(x):
        return torch.rand(x, dtype=torch.float32, device=device)

    images = rand((b, c, h, w))
    samples = 2.5 * rand((b, s, s2, 2)) - 1.25
    weights = rand((b, hd, s, s2))

    # test forward pass
    custom_time = 0
    torch_time = 0
    with torch.no_grad():
        for _ in trange(500):
            torch.cuda.synchronize()
            start = time()
            custom = grid_sample_dot(images, samples)
            torch.cuda.synchronize()
            custom_time += time() - start
            start = time()
            original = torch.nn.functional.grid_sample(
                images,
                samples,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            torch.cuda.synchronize()
            torch_time += time() - start
    assert torch.allclose(custom, original)
    print(f"custom time: {custom_time}")
    print(f"torch time: {torch_time}")

    a = 1

    forward_time = 0
    backward_time = 0
    for i in trange(100):
        images_expected = images.clone().requires_grad_(True)
        weights_expected = weights.clone().requires_grad_(True)
        torch.cuda.synchronize()
        start = time()
        result_expected = fused_grid_sum_torch(
            images_expected, samples, weights_expected
        )
        torch.cuda.synchronize()
        forward_time += time() - start

        start = time()
        result_expected.sum().backward()
        torch.cuda.synchronize()
        backward_time += time() - start

    print(f"forward time: {forward_time}")
    print(f"backward time: {backward_time}")

    forward_time = 0
    backward_time = 0
    for i in trange(100):
        images_actual = images.clone().requires_grad_(True)
        weights_actual = weights.clone().requires_grad_(True)
        torch.cuda.synchronize()
        start = time()
        result_actual = fused_grid_sum(images_actual, samples, weights_actual)
        torch.cuda.synchronize()
        forward_time += time() - start

        start = time()
        result_actual.sum().backward()
        torch.cuda.synchronize()
        backward_time += time() - start

    print(f"forward time: {forward_time}")
    print(f"backward time: {backward_time}")

    assert torch.allclose(result_actual, result_expected, atol=1e-4)
    print(f"result max diff: {(result_actual - result_expected).abs().max()}")
    assert torch.allclose(images_actual.grad, images_expected.grad, atol=1e-4)
    print(
        f"image grad max diff: {(images_actual.grad - images_expected.grad).abs().max()}"
    )
    assert torch.allclose(weights_actual.grad, weights_expected.grad, atol=1e-4)
    print(
        f"weight grad max diff: {(weights_actual.grad - weights_expected.grad).abs().max()}"
    )
