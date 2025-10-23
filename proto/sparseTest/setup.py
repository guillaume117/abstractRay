from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_conv',
    ext_modules=[
        CUDAExtension(
            name='sparse_conv',
            sources=['sparse_conv.cpp', 'sparse_conv.cu'],  # Uniquement une fois chaque fichier
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '--expt-relaxed-constexpr',
                    '-O2',
                    '-gencode=arch=compute_70,code=sm_70',  # Ajustez selon votre GPU
                    '-gencode=arch=compute_89,code=sm_89'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
