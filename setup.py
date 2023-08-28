from os import path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="torchdraw_cuda",
    packages=["torchdraw_cuda"],
    install_requires=["torch", "jaxtyping"],
    ext_modules=[
        CUDAExtension(
            name="torchdraw_cuda._cuda",
            sources=[
                "src/extension.cu",
                "src/points.cu",
                "src/lines.cu",
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
