from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_sample_ops",
    packages=["fused_sample_ops"],
    install_requires=["torch", "jaxtyping"],
    ext_modules=[
        CUDAExtension(
            name="fused_sample_ops._cuda",
            sources=[
                "src/extension.cu",
                "src/sample_dot.cu",
                "src/sample_sum.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
