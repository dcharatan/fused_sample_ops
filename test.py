from time import time

import torch
from jaxtyping import install_import_hook
from tqdm import trange

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from fused_grid_sum import fused_grid_sum, sample_dot, sample_dot_torch
    from tests.fused_grid_sum_torch import fused_grid_sum_torch


if __name__ == "__main__":
    device = torch.device("cuda:0")

    NUM_TESTS = 100
    B = 8
    C_QUERIES = 128
    H = 128
    W = 256
    NUM_OCTAVES = 6
    Q = 256
    D = 123

    C_IMAGES = C_QUERIES + 2 * NUM_OCTAVES

    def rand(x):
        return torch.rand(x, dtype=torch.float64, device=device)

    images = rand((B, C_IMAGES, H, W))
    samples = 2.5 * rand((B, Q, D, 2)) - 1.25
    queries = rand((B, Q, C_QUERIES))
    depths = rand((B, Q, D))

    # test forward pass
    forward_time_fused = 0
    backward_time_fused = 0
    forward_time_torch = 0
    backward_time_torch = 0
    for _ in trange(NUM_TESTS):
        torch.cuda.synchronize()
        images_fused = images.clone().requires_grad_(True)
        samples_fused = samples.clone().requires_grad_(True)
        queries_fused = queries.clone().requires_grad_(True)
        depths_fused = depths.clone().requires_grad_(True)
        start = time()
        result_fused = sample_dot(
            images_fused,
            samples_fused,
            queries_fused,
            depths_fused,
            NUM_OCTAVES,
        )
        torch.cuda.synchronize()
        forward_time_fused += time() - start
        result_fused.sum().backward()
        torch.cuda.synchronize()
        backward_time_fused += time() - start

        torch.cuda.synchronize()
        images_torch = images.clone().requires_grad_(True)
        samples_torch = samples.clone().requires_grad_(True)
        queries_torch = queries.clone().requires_grad_(True)
        depths_torch = depths.clone().requires_grad_(True)
        start = time()
        result_torch = sample_dot_torch(
            images_torch,
            samples_torch,
            queries_torch,
            depths_torch,
            NUM_OCTAVES,
        )
        torch.cuda.synchronize()
        forward_time_torch += time() - start
        result_torch.sum().backward()
        torch.cuda.synchronize()
        backward_time_torch += time() - start

        assert torch.allclose(result_fused, result_torch)
        assert torch.allclose(images_fused.grad, images_torch.grad, atol=5e-5)
        assert torch.allclose(samples_fused.grad, samples_torch.grad, atol=5e-5)
        assert torch.allclose(queries_fused.grad, queries_torch.grad, atol=5e-5)
        assert torch.allclose(depths_fused.grad, depths_torch.grad, atol=5e-5)

    print(f"forward (fused): {forward_time_fused}")
    print(f"forward (torch): {forward_time_torch}")
    print(f"backward (fused): {backward_time_fused}")
    print(f"backward (torch): {backward_time_torch}")

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
