from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="torchdraw_cuda",
    ext_modules=[
        CUDAExtension(
            "torchdraw_cuda",
            [
                "src/extension.cu",
                "src/points.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
