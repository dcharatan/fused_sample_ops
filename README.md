# fused_grid_ops

This repository contains fused CUDA kernels for the following operations:

```python
def sample_sum_torch(
    images: Float[Tensor, "batch channel height width"],
    samples: Float[Tensor, "batch query depth 2"],
    weights: Float[Tensor, "batch head query depth"],
) -> Float[Tensor, "batch head query channel"]:
    grid_samples = F.grid_sample(
        images,
        samples,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return einsum(grid_samples, weights, "b c q d, b hd q d -> b hd q c")
```

```python
def sample_dot_torch(
    images: Float[Tensor, "batch channel height width"],
    samples: Float[Tensor, "batch query depth 2"],
    queries: Float[Tensor, "batch head query channel-2*num_octaves"],
    depths: Float[Tensor, "batch query depth"],
    num_octaves: int,
) -> Float[Tensor, "batch head query depth"]:
    samples = F.grid_sample(
        images,
        samples,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    # Generate positional encoding frequencies and phases.
    frequencies = torch.arange(num_octaves, dtype=queries.dtype, device=queries.device)
    frequencies = 2 * torch.pi * 2**frequencies
    phases = torch.tensor([0, 1], dtype=queries.dtype, device=queries.device)
    phases = 0.5 * torch.pi * phases

    # Positionally encode the depths.
    _, _, d = depths.shape
    _, hd, _, _ = queries.shape
    frequencies = repeat(frequencies, "f -> () (f p) () ()", p=2)
    phases = repeat(phases, "p -> () (f p) () ()", f=num_octaves)
    depths = rearrange(depths, "b q d -> b () q d")
    depths = torch.sin(depths * frequencies + phases)
    depths = repeat(depths, "b c q d -> b hd c q d", hd=hd)

    # Concatenate the positionally encoded depths onto the queries.
    queries = repeat(queries, "b hd q c -> b hd c q d", d=d)
    queries = torch.cat((queries, depths), dim=2)

    return einsum(samples, queries, "b c q d, b hd c q d -> b hd q d")

```

## Compilation

First, set up your virtual environment. Make sure the PyTorch wheel you use matches your local CUDA version.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -r requirements.txt
```

### Development: Using CMake and VS Code

Inside VS Code, run `CMake: Configure` and then `CMake: Build`. You may want to do `CMake: Select Variant` first. This will create `build/_cuda.so`.

### Production: Using `setup.py`

To install this package, do `python3 setup.py install` from the project root directory. To build without installing, run `python3 setup.py develop`. This will create `_cuda.<stuff>.so`.

### Importing `fused_grid_ops`

Once the `.so` file is created in the `fused_grid_ops` directory (using CMake, you'll have to move it there manually), you can import `fused_grid_ops` and use the provided Python wrappers. These wrappers have `jaxtyping` annotations that ensure that the CUDA code is called correctly.
