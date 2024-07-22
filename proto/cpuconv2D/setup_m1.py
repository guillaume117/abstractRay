from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='sparse_conv2d',
    ext_modules=[
        CppExtension(
            'sparse_conv2d',
            ['sparse_conv2d.cpp'],
            extra_compile_args=['-Xpreprocessor', '-fopenmp', '-I/usr/local/include', '-O3', '-std=c++17'],
            extra_link_args=['-L/usr/local/lib', '-lomp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }
)
