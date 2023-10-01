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

    image = torch.ones((b, c, h, w), dtype=torch.float32, device=device)
    samples = 0 * torch.ones((b, s, s2, 2), dtype=torch.float32, device=device)
    weights = torch.ones((b, hd, s, s2), dtype=torch.float32, device=device)

    result = fused_grid_sum_forward(image, samples, weights)

    # mini benchmark

    from tqdm import trange
    import torch.nn.functional as F
    from einops import einsum

    for _ in trange(1000):
        torch.cuda.synchronize(device)
        result = fused_grid_sum_forward(image, samples, weights)
        torch.cuda.synchronize(device)

    for _ in trange(1000):
        torch.cuda.synchronize(device)
        grid_samples = F.grid_sample(
            image,
            samples,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        result = einsum(grid_samples, weights, "b c s s2, b hd s s2 -> b hd s c")
        torch.cuda.synchronize(device)

    a = 1
    a = 1
