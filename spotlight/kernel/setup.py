from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='lsh_kernel_cuda',
    ext_modules=[
        CUDAExtension(
            'attn_k4_q28',
            ['attn_k4_q28.cu'],
            extra_compile_args={'nvcc': ['-O3', '-Xptxas', '-v']}
            ),
        CUDAExtension(
            'packbits',
            ['packbits.cu'],
            extra_compile_args={'nvcc': ['-O3', '-Xptxas', '-v']}
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)