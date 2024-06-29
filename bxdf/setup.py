from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
import platform

__version__ = "0.2.0"
cxx_std=11

ext_modules = [
    Pybind11Extension("vol_loader",
        ["vol_loader/vol2numpy.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args= ['-g', '-O3' if platform.system() == "Linux" else '-O2', '-fopenmp'],
        ),
]

setup(
    name="vol_loader",
    version=__version__,
    author="Qianyue He",
    description="Volume grid loader",
    include_dirs="E:\\eigen-3.4.0",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
    python_requires=">=3.7",
)
