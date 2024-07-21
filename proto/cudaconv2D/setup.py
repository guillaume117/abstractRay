from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch
from torch.utils.cpp_extension import CUDAExtension

ext_modules = [
    CUDAExtension(
        "sparse_conv2d",
        ["sparse_conv2d.cu"],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
    )
]

setup(
    name="sparse_conv2d",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
