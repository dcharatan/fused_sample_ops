# torchdraw_cuda

This repository contains unoptimized CUDA functions for drawing points and lines. I mainly created it to familiarize myself with the workflow for creating CUDA PyTorch extensions, although it runs much faster than the equivalent PyTorch functions.

## Optimizations

For small numbers of lines, `render_lines` is relatively fast on my workstation's 3090 Ti. For example, it renders 64,000,000 samples with 256 lines in 0.157 seconds, which is fast enough for non-real-time visualization. However, for larger numbers of lines, it becomes unacceptably slow. For example, rendering 16,384 lines takes 9.382 seconds. To improve this, one would probably first parallelize over lines to "scatter" them onto image tiles, then parallelize over samples and only consider lines in the corresponding tile.

## Compilation

### Development: Using CMake and VS Code

Inside VS Code, run `CMake: Configure` and then `CMake: Build`. You may want to do `CMake: Select Variant` first. This will create `build/_cuda.so`.

### Production: Using `setup.py`

To install this package, do `python3 setup.py install` from the project root directory. To build without installing, run `python3 setup.py develop`. This will create `_cuda.<stuff>.so`.

### Importing `torchdraw_cuda`

Once the `.so` file is created in the `torchdraw_cuda` directory (using CMake, you'll have to move it there manually), you can import `torchdraw_cuda` and use the provided Python wrappers. These wrappers have `jaxtyping` annotations that ensure that the CUDA code is called correctly.
