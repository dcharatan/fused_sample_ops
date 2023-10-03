from time import time

import torch
from jaxtyping import install_import_hook
from tqdm import trange

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from fused_sample_ops.sample_dot_fused import sample_dot_fused
    from fused_sample_ops.sample_dot_torch import sample_dot_torch
    from fused_sample_ops.sample_sum_fused import sample_sum_fused
    from fused_sample_ops.sample_sum_torch import sample_sum_torch


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_printoptions(4, sci_mode=False)

    NUM_TESTS = 100
    HD = 4
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

    # Test sample_dot.
    images = rand((B, C_IMAGES, H, W))
    samples = 2.5 * rand((B, Q, D, 2)) - 1.25
    queries = rand((B, HD, Q, C_QUERIES))
    depths = rand((B, Q, D))

    forward_time_fused = 0
    backward_time_fused = 0
    forward_time_torch = 0
    backward_time_torch = 0

    for _ in trange(NUM_TESTS, desc="sample_dot"):
        torch.cuda.synchronize()
        images_fused = images.clone().requires_grad_(True)
        samples_fused = samples.clone().requires_grad_(True)
        queries_fused = queries.clone().requires_grad_(True)
        depths_fused = depths.clone().requires_grad_(True)
        start = time()
        result_fused = sample_dot_fused(
            images_fused,
            samples_fused,
            queries_fused,
            depths_fused,
            NUM_OCTAVES,
        )
        torch.cuda.synchronize()
        forward_time_fused += time() - start
        start = time()
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
        start = time()
        result_torch.sum().backward()
        torch.cuda.synchronize()
        backward_time_torch += time() - start

        assert torch.allclose(result_fused, result_torch)
        assert torch.allclose(images_fused.grad, images_torch.grad, atol=5e-5)
        assert torch.allclose(samples_fused.grad, samples_torch.grad, atol=5e-5)
        assert torch.allclose(queries_fused.grad, queries_torch.grad, atol=5e-5)
        assert torch.allclose(depths_fused.grad, depths_torch.grad, atol=5e-5)

    print("sample_dot:")
    print(f"forward (fused): {forward_time_fused}")
    print(f"forward (torch): {forward_time_torch}")
    print(f"backward (fused): {backward_time_fused}")
    print(f"backward (torch): {backward_time_torch}")

    # Test sample_sum.
    images = rand((B, C_IMAGES, H, W))
    samples = 2.5 * rand((B, Q, D, 2)) - 1.25
    weights = rand((B, HD, Q, D))

    forward_time_fused = 0
    backward_time_fused = 0
    forward_time_torch = 0
    backward_time_torch = 0

    for _ in trange(NUM_TESTS, desc="sample_sum"):
        torch.cuda.synchronize()
        images_fused = images.clone().requires_grad_(True)
        samples_fused = samples.clone().requires_grad_(True)
        weights_fused = weights.clone().requires_grad_(True)
        start = time()
        result_fused = sample_sum_fused(images_fused, samples_fused, weights_fused)
        torch.cuda.synchronize()
        forward_time_fused += time() - start
        start = time()
        result_fused.sum().backward()
        torch.cuda.synchronize()
        backward_time_fused += time() - start

        torch.cuda.synchronize()
        images_torch = images.clone().requires_grad_(True)
        samples_torch = samples.clone().requires_grad_(True)
        weights_torch = weights.clone().requires_grad_(True)
        start = time()
        result_torch = sample_sum_torch(images_torch, samples_torch, weights_torch)
        torch.cuda.synchronize()
        forward_time_torch += time() - start
        start = time()
        result_torch.sum().backward()
        torch.cuda.synchronize()
        backward_time_torch += time() - start

        assert torch.allclose(result_fused, result_torch)
        assert torch.allclose(images_fused.grad, images_torch.grad, atol=5e-5)
        assert torch.allclose(samples_fused.grad, samples_torch.grad, atol=5e-5)
        assert torch.allclose(weights_fused.grad, weights_torch.grad, atol=5e-5)

    print("sample_sum:")
    print(f"forward (fused): {forward_time_fused}")
    print(f"forward (torch): {forward_time_torch}")
    print(f"backward (fused): {backward_time_fused}")
    print(f"backward (torch): {backward_time_torch}")
