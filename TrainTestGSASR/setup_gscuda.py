from setuptools importsetup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


file_path = "utils/gs_cuda_dmax"

setup(
    name="gscuda", 
    ext_modules=[
        CUDAExtension(
            name="gscuda", 
            sources=[
                os.path.join(file_path, "gswrapper.cpp"),
                os.path.join(file_path, "gs.cu")
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)