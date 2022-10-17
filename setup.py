from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='fcpr',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='fcpr.ext',
            sources=[
                'fcpr/extensions/extra/cloud/cloud.cpp',
                'fcpr/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'fcpr/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'fcpr/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'fcpr/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'fcpr/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
