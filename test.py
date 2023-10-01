import torch
from jaxtyping import install_import_hook

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from fused_grid_sum import fused_grid_sum_forward

if __name__ == "__main__":
    device = torch.device("cuda:0")

    b = 8
    c = 128
    h = 129
    w = 130
    s = 131
    s2 = 132
    hd = 8

    image = torch.rand((b, c, h, w), dtype=torch.float32, device=device)
    samples = 2 * torch.rand((b, s, s2, 2), dtype=torch.float32, device=device) - 1
    weights = torch.rand((b, hd, s, s2), dtype=torch.float32, device=device)

    result = fused_grid_sum_forward(image, samples, weights)

    # mini benchmark

    from tqdm import trange, tqdm
    import torch.nn.functional as F
    from einops import einsum
    from itertools import permutations

    for _ in trange(1):
        torch.cuda.synchronize(device)
        result = fused_grid_sum_forward(image, samples, weights)
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
        ours = fused_grid_sum_forward(
            image, samples, torch.ones((1, 1, 1, 1), dtype=torch.float32, device=device)
        )
        return original, ours

    a = 1
    a = 1
