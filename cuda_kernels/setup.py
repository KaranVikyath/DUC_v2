"""
Build script for the fused pseudo-completion CUDA extension.

Usage:
    cd cuda_kernels
    python setup.py install          # install system-wide
    python setup.py build_ext --inplace  # build in-place for development
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pseudo_completion_cuda",
    ext_modules=[
        CUDAExtension(
            "pseudo_completion_cuda",
            ["pseudo_completion.cu"],
            extra_compile_args={
                "nvcc": [
                    "--allow-unsupported-compiler",
                    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
