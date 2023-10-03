from os import path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_grid_sum",
    packages=["fused_grid_sum"],
    install_requires=["torch", "jaxtyping"],
    ext_modules=[
        CUDAExtension(
            name="fused_grid_sum._cuda",
            sources=[
                "src/extension.cu",
                "src/fused_grid_sum.cu",
                "src/sample_dot.cu",
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
