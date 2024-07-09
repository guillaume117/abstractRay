from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension


omp_include_path ="/usr/lib/llvm-14/lib/clang/14.0.0/include/omp.h"


ext_modules = [
    CppExtension(
        "sparse_conv2d",
        ["sparse_conv2d.cpp"],
        include_dirs=[torch.utils.cpp_extension.include_paths()[0]],
        extra_compile_args=['-O3', '-std=c++17'],
          # Assurez-vous de lier avec OpenMP
    ),
]

setup(
    name="sparse_conv2d",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
