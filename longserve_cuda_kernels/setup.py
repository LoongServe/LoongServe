from setuptools import setup
from torch.utils import cpp_extension
import os, sys

__version__ = "0.0.1"

CUDA_HOME = os.getenv("CUDA_HOME", None)
if CUDA_HOME == None:
    print("Error: CUDA_HOME not set")
    sys.exit(1)

CONDA_PREFIX = os.getenv("CONDA_PREFIX", None)
if CONDA_PREFIX == None:
    print("Error: CONDA_PREFIX not set")
    sys.exit(1)

include_dirs = [
    os.path.join(CUDA_HOME, "include"),
]

library_dirs = [
    os.path.join(CONDA_PREFIX, "lib"),
]

ext_modules = [
    cpp_extension.CUDAExtension(
        "longserve_cuda_kernels",
        [
            "src/entrypoints.cpp",
            "src/rms_norm.cu",
            "src/silu_and_mul.cu",
            "src/rotary_emb.cu",
            "src/flash_decoding_stage1.cu",
            "src/flash_decoding_stage2.cu",
        ],
		include_dirs=include_dirs,
		library_dirs=library_dirs,
        extra_compile_args={
            'cxx': ['-O2'],
            'nvcc': ['-O3', '--use_fast_math']
        }
    ),
]

setup(
    name="longserve_cuda_kernels",
    version=__version__,
    author="Bingyang Wu, Shengyu Liu",
    author_email="",
    url="",
    description="Some CUDA kernels for LongServe",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    zip_safe=False,
    python_requires=">=3.7",
)
