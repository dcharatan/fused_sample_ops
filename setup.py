from os import path

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
            extra_compile_args={
                "nvcc": [
                    "-I"
                    + path.join(
                        path.dirname(path.abspath(__file__)), "third_party/glm/"
                    )
                ]
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
