import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

LIBRARY_NAME = "trec"


def get_extensions():
    DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
    if DEBUG_MODE:
        print("Building in debug mode")

    assert (torch.cuda.is_available() and CUDA_HOME is not None)

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-fopenmp",
            "-O3",
            "-DNDEBUG" if not DEBUG_MODE else "-g",
            "-fdiagnostics-color=always",
            "-Wall",
            "-Wextra",
        ],
        "nvcc": [
            "-O3",
            "-DNDEBUG" if not DEBUG_MODE else "-G",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            "--use_fast_math",  # ! Assessments needed
        ],
    }
    if DEBUG_MODE:
        extra_link_args += ["-Og", "-g", "-lto"]

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, LIBRARY_NAME, "csrc")

    sources = glob.glob(os.path.join(
        extensions_dir, "**", "*.cpp"), recursive=True)
    cuda_sources = glob.glob(os.path.join(
        extensions_dir, "**", "*.cu"), recursive=True)
    sources += cuda_sources

    define_macros = [("WITH_CUDA", None)]

    CC = os.environ.get("CC", None)
    if CC is not None:
        extra_compile_args["nvcc"].append(f"-ccbin={CC}")

    include_dirs = [extensions_dir]

    ext_modules = [
        CUDAExtension(
            f"{LIBRARY_NAME}._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name=LIBRARY_NAME,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="spatially efficient new form of convolution",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    cmdclass={"build_ext": BuildExtension},
)
