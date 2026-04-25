"""
Build script for DeLUCA CUDA extensions.

Usage:
    cd cuda_kernels
    python setup.py install          # install system-wide
    python setup.py build_ext --inplace  # build in-place for development
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_FLAGS = [
    "--allow-unsupported-compiler",
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
]

setup(
    name="deluca_cuda_extensions",
    ext_modules=[
        CUDAExtension(
            "pseudo_completion_cuda",
            ["pseudo_completion.cu"],
            extra_compile_args={"nvcc": NVCC_FLAGS},
        ),
        CUDAExtension(
            "cfs_solver_cuda",
            ["cfs_solver.cu"],
            extra_compile_args={"nvcc": NVCC_FLAGS},
            libraries=["cusolver"],
        ),
        CUDAExtension(
            "masked_loss_cuda",
            ["masked_loss.cu"],
            extra_compile_args={"nvcc": NVCC_FLAGS},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
