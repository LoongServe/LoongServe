from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "longserve_c_scheduler",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
        extra_compile_args=['-static', '-O2']
    ),
]

setup(
    name="longserve_c_scheduler",
    version=__version__,
    author="Bingyang Wu, Shengyu Liu",
    author_email="",
    url="",
    description="Some C++ implementations for LongServe scheduler",
    long_description="",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7",
)
