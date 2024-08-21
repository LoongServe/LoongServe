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
    cpp_extension.CppExtension(
        "rnccl",
        ["src/main.cpp"],
		include_dirs=include_dirs,
		library_dirs=library_dirs,
		extra_compile_args=['-static']
    ),
]

setup(
    name="RNCCL",
    version=__version__,
    author="Bingyang Wu, Shengyu Liu",
    author_email="",
    url="",
    description="Raw NCCL Bindings for Python",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
    python_requires=">=3.7",
)
