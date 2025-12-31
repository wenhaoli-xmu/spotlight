import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

capability = torch.cuda.get_device_capability()
major, minor = capability
arch = f"sm_{major}{minor}"

cxx_args = ['-O3', '-std=c++17']
nvcc_args = [
    '-O3',
    '-std=c++17',
    f'-arch={arch}',
    '--use_fast_math',
    '--expt-relaxed-constexpr',
    '--threads=4',
    '-Xptxas', '-v',
]

setup(
    name='lsh_kernel_cuda',
    ext_modules=[
        CUDAExtension(
            'attn_k8_q32',
            ['attn_k8_q32.cu'],
            extra_compile_args={
                'cxx': cxx_args, 
                'nvcc': nvcc_args
            }
        ),
        CUDAExtension(
            'packbits',
            ['packbits.cu'],
            extra_compile_args={
                'cxx': cxx_args, 
                'nvcc': nvcc_args
            }
        ),  
        CUDAExtension(
            'lru_cache_kernel',
            ['lru_cache_kernel.cu'],
            extra_compile_args={
                'cxx': cxx_args, 
                'nvcc': nvcc_args
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)