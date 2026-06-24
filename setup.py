from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_layernorm_backend',
    ext_modules=[
        CUDAExtension(
            name='my_layernorm_backend',
            sources=['layernorm_wrapper.cpp', 'layernorm_kernel.hip'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
