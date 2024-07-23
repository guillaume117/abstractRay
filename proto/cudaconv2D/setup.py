from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    name='sparse_conv2d',
    ext_modules=[
        CUDAExtension(
            name='sparse_conv2d',
            sources=['sparse_conv2d.cpp', 'sparse_conv2d_cuda.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
