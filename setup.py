import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

LIBRARY_NAME = "trec"
EXTENSION_NAME = f"{LIBRARY_NAME}._C"
EXTENSION_DIR = os.path.join(os.path.dirname(os.path.curdir), LIBRARY_NAME, "csrc")
INCLUDE_DIRS = [EXTENSION_DIR]
DEFINE_MACROS = [("WITH_CUDA", None)]

DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
PROFILE_MODE = os.getenv("PROFILE", "0") == "1"
RELEASE_MODE = os.getenv("RELEASE", "0") == "1"
PRINT_FLAGS = os.getenv("PRINT", "0") == "1"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

class MyBuildExtension(BuildExtension):
    """
    Specify ``-isystem`` flag for ``include_dirs``, since :class:`BuildExtension`
    generates ``-I`` flag, causing warnings to be reported in torch headers.
    """

    def build_extensions(self):
        for ext in self.extensions:
            compile_args = ext.extra_compile_args
            for include_dir in ext.include_dirs:
                if "cxx" in compile_args:
                    compile_args["cxx"] += ["-isystem", f"{include_dir}"]
                if "nvcc" in ext.extra_compile_args:
                    compile_args["nvcc"] += ["-isystem", f"{include_dir}"]
        super().build_extensions()


def get_extensions():
    if DEBUG_MODE:
        print(f"{RED}Building in debug mode{RESET}")

    assert torch.cuda.is_available() and CUDA_HOME is not None, f"{RED}CUDA is not available.{RESET}"

    extra_compile_args = {
        "cxx": [
            "-std=c++20",
        ],
        "nvcc": [
            "-std=c++20",
            # CUDA specific
            "-DCUDA_HAS_FP16=1",  # ? not sure if this is needed
            "--expt-extended-lambda",
            "--use_fast_math",
            # common cxx flags
            "-Xcompiler",
            "-Wall",
            "-Xcompiler",
            "-Wextra",
        ],
    }
    extra_link_args = []

    if RELEASE_MODE:
        extra_compile_args["cxx"] += ["-O3"]
        extra_compile_args["nvcc"] += ["-O3"]
    elif DEBUG_MODE:
        extra_compile_args["cxx"] += ["-g", "-Og"]
        extra_compile_args["nvcc"] += ["-g", "-G"]  # suppress device opt
        extra_link_args += ["-g"]
    else:  # default to PROFILE_MODE
        extra_compile_args["cxx"] += ["-O3", "-g", "-pg", "-lineinfo"]
        extra_compile_args["nvcc"] += ["-O3", "-g", "-pg"]

    if not PRINT_FLAGS:
        extra_compile_args["cxx"] += ["-DNDEBUG"]
        extra_compile_args["nvcc"] += ["-DNDEBUG"]

    sources = glob.glob(
        os.path.join(EXTENSION_DIR, "**", "*.cpp"), recursive=True
    ) + glob.glob(os.path.join(EXTENSION_DIR, "**", "*.cu"), recursive=True)

    CC = os.environ.get("CC", None)
    if CC is not None:
        extra_compile_args["nvcc"].append(f"-ccbin={CC}")

    ext_modules = [
        CUDAExtension(
            EXTENSION_NAME,
            sources,
            include_dirs=INCLUDE_DIRS,
            define_macros=DEFINE_MACROS,
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
    cmdclass={"build_ext": MyBuildExtension},
)
