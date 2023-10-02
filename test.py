import torch
from jaxtyping import install_import_hook

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from fused_grid_sum import fused_grid_sum
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

    images_expected = images.clone().requires_grad_(True)
    weights_expected = weights.clone().requires_grad_(True)
    result_expected = fused_grid_sum_torch(images_expected, samples, weights_expected)
    result_expected.sum().backward()

    images_actual = images.clone().requires_grad_(True)
    weights_actual = weights.clone().requires_grad_(True)
    result_actual = fused_grid_sum(images_actual, samples, weights_actual)
    result_actual.sum().backward()

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

    # stuff below this line is currently junk

    for _ in trange(1):
        torch.cuda.synchronize(device)
        result = fused_grid_sum(image, samples, weights)
        torch.cuda.synchronize(device)

    for _ in trange(1):
        torch.cuda.synchronize(device)
        grid_samples = F.grid_sample(
            image,
            samples,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        result2 = einsum(grid_samples, weights, "b c s s2, b hd s s2 -> b hd s c")
        torch.cuda.synchronize(device)

    def compare(samples):
        image = torch.zeros((1, 1, 2, 2), dtype=torch.float32, device=device)
        image[..., 0, 0] = 1
        image[..., 0, 1] = 10
        image[..., 1, 0] = 100
        image[..., 1, 1] = 1000
        samples = torch.tensor(samples, device=device, dtype=torch.float32)[
            None, None, None
        ]

        original = F.grid_sample(
            image,
            samples,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        ours = fused_grid_sum(
            image, samples, torch.ones((1, 1, 1, 1), dtype=torch.float32, device=device)
        )
        return original, ours

    a = 1
    a = 1
