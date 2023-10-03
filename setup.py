from os import path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_grid_ops",
    packages=["fused_grid_ops"],
    install_requires=["torch", "jaxtyping"],
    ext_modules=[
        CUDAExtension(
            name="fused_grid_ops._cuda",
            sources=[
                "src/extension.cu",
                "src/sample_dot.cu",
                "src/sample_sum.cu",
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
