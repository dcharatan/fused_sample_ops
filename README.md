# fused_grid_sum

This repository contains unoptimized but fused CUDA code for the following operation:

```
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
```

The kernel fusion means that the result of `grid_sample` doesn't need to be saved, dramatically reducing GPU memory usage.

## Compilation

### Development: Using CMake and VS Code

Inside VS Code, run `CMake: Configure` and then `CMake: Build`. You may want to do `CMake: Select Variant` first. This will create `build/_cuda.so`.

### Production: Using `setup.py`

To install this package, do `python3 setup.py install` from the project root directory. To build without installing, run `python3 setup.py develop`. This will create `_cuda.<stuff>.so`.

### Importing `fused_grid_sum`

Once the `.so` file is created in the `fused_grid_sum` directory (using CMake, you'll have to move it there manually), you can import `fused_grid_sum` and use the provided Python wrappers. These wrappers have `jaxtyping` annotations that ensure that the CUDA code is called correctly.
