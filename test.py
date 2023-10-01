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
    h = 128
    w = 128
    s = 128
    s2 = 128
    hd = 8

    image = torch.ones((b, c, h, w), dtype=torch.float32, device=device)
    samples = torch.ones((b, s, s2, 2), dtype=torch.float32, device=device)
    weights = torch.ones((b, s, hd, s2), dtype=torch.float32, device=device)

    result = fused_grid_sum_forward(image, samples, weights)

    a = 1
    a = 1
