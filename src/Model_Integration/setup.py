from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='custom_conv',
    ext_modules=[
        CUDAExtension('custom_conv', [
            'convolution_regular.cu',
            'convoluion_inline_ptx.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)

