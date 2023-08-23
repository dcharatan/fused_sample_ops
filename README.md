# torchdraw_cuda

## Compilation

### Development: Using CMake and VS Code

Inside VS Code, run `CMake: Configure` and then `CMake: Build`. You may want to do `CMake: Select Variant` first. This will create `build/torchdraw_cuda.so`.

### Production: Using `setup.py`

To install this package, do `pip install .` from the project root directory. To build without installing, run `python3 setup.py develop`. This will create `torchdraw_cuda.<stuff>.so`.

### Importing `torchdraw_cuda`

If you've created a `.so` file manually using either compilation method, add the folder containing it to `sys.path`. Then, you can import `torchdraw_cuda` as follows:

```
import torch
import torchdraw_cuda
```

Note that PyTorch must be imported first, since it loads necessary libraries (e.g., `libc10.so`)!
