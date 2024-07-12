from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension

ext_modules = [
    CppExtension(
        "custom_conv",
        ["sparse_conv2d_custom.cpp"],
        include_dirs=[torch.utils.cpp_extension.include_paths()[0]],
        extra_compile_args=['-fopenmp','-O3', '-std=c++17'],
        extra_link_args=['-fopenmp']
    ),
]

setup(
    name="custom_conv",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)