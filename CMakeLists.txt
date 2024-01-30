# Set the minimum version of CMake that can be used
cmake_minimum_required(VERSION 3.1)

# ! set `export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"` before running cmake
# see https://github.com/pytorch/pytorch/issues/113948

# Set the project name
project(ProfilingProject)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CUDA_TOOLKIT_ROOT_DIR "/home/lizhifei/miniconda3/envs/trec")
set(CUDA_INCLUDE_DIRS "/home/lizhifei/miniconda3/envs/trec/include")

set(CMAKE_CUDA_COMPILER "/home/lizhifei/miniconda3/envs/trec/bin/nvcc")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
# # Ensure CMake can find PyTorch and other required packages in your Conda environment
list(APPEND CMAKE_PREFIX_PATH "/home/lizhifei/miniconda3/envs/trec")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# SET(GCC_COVERAGE_LINK_FLAGS    "-lrt")


add_compile_options(-lnvToolsExt -export-dynamic -fPIC -finstrument-functions -ldl -lnvToolsExt)

set(Torch_DIR /home/lizhifei/miniconda3/envs/trec/lib/python3.12/site-packages/torch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(prof prof.cpp 
    inst_nvtx.o
    /home/lizhifei/TREC-Artifact/build/temp.linux-x86_64-cpython-312/home/lizhifei/TREC-Artifact/trec_op/ConvDeepReuseBackward/convDR_backward.o
    /home/lizhifei/TREC-Artifact/build/temp.linux-x86_64-cpython-312/home/lizhifei/TREC-Artifact/trec_op/ConvDeepReuseForward/convDR_forward.o
)

set_source_files_properties(
    inst_nvtx.o
    /home/lizhifei/TREC-Artifact/build/temp.linux-x86_64-cpython-312/home/lizhifei/TREC-Artifact/trec_op/ConvDeepReuseBackward/convDR_backward.o
    /home/lizhifei/TREC-Artifact/build/temp.linux-x86_64-cpython-312/home/lizhifei/TREC-Artifact/trec_op/ConvDeepReuseForward/convDR_forward.o
    PROPERTIES
    EXTERNAL_OBJECT true
    GENERATED true
)

# Include directories for PyTorch and your own include directory
target_include_directories(prof PRIVATE
    ${TORCH_INCLUDE_DIRS}
    /home/lizhifei/miniconda3/envs/trec/include
    /home/lizhifei/TREC-Artifact/trec_op
)

set(PYTHON_INCLUDE_DIRS "/home/lizhifei/miniconda3/envs/trec/include/python3.12")
include_directories(/home/lizhifei/miniconda3/envs/trec/include/python3.12)

# Find Pybind11
find_package(pybind11 REQUIRED)

# Link against Pybind11
target_link_libraries(prof pybind11::pybind11)


find_package(Python REQUIRED)

target_link_libraries(prof ${PYTHON_LIBRARIES})

# Link against the PyTorch libraries
target_link_libraries(prof "${TORCH_LIBRARIES}")

# This is required for PyTorch
set_property(TARGET prof PROPERTY CXX_STANDARD 17)

# If you need OpenMP (it was in your original compile command)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(prof OpenMP::OpenMP_CXX)
endif()
