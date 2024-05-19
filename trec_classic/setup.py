import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"


def get_extensions():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # c++ source code directory
    extensions_dir = os.path.join(this_dir, "trec_op")

    sources = glob.glob(os.path.join(
        extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    extension = CppExtension

    extra_compile_args = {"cxx": ["-fopenmp"]}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
        ]

        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "trec_op",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="trec_op",
    # version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.6",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
