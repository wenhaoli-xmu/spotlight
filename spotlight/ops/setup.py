from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hamming_ops',
    ext_modules=[
        CUDAExtension(
            name='hamming_ops',
            sources=['hamming_ops.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)