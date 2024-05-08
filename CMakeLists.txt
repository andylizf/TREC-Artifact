cmake_minimum_required(VERSION 3.1)

# ! set `export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"` before running cmake
# see https://github.com/pytorch/pytorch/issues/113948

project(ProfilingProject)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(USE_NVTX "Enable NVTX profiling" OFF)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set CUDA related paths
set(CUDA_TOOLKIT_ROOT_DIR "/home/lizhifei/miniconda3/envs/trec2")
set(CUDA_INCLUDE_DIRS "/home/lizhifei/miniconda3/envs/trec2/include")
set(CMAKE_CUDA_COMPILER "/home/lizhifei/miniconda3/envs/trec2/bin/nvcc")

# Ensure CMake can find PyTorch and other required packages in your Conda environment
list(APPEND CMAKE_PREFIX_PATH "/home/lizhifei/miniconda3/envs/trec2")

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find PyTorch
set(Torch_DIR /home/lizhifei/miniconda3/envs/trec2/lib/python3.12/site-packages/torch/share/cmake/Torch)
find_package(Torch REQUIRED)

# Find Pybind11
find_package(pybind11 REQUIRED)

if(USE_NVTX)
    # Compile inst_nvtx.cpp as a separate object
    add_library(inst_nvtx OBJECT inst_nvtx.cpp)
    target_include_directories(inst_nvtx PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_options(inst_nvtx PRIVATE -fPIC -export-dynamic)

    # Create the executable
    add_executable(prof prof.cpp 
        $<TARGET_OBJECTS:inst_nvtx>
        /home/lizhifei/TREC-Artifact/build/temp.linux-x86_64-cpython-312/home/lizhifei/TREC-Artifact/trec_op/ConvDeepReuseBackward/convDR_backward.o
        /home/lizhifei/TREC-Artifact/build/temp.linux-x86_64-cpython-312/home/lizhifei/TREC-Artifact/trec_op/ConvDeepReuseForward/convDR_forward.o
    )

    # Link libraries to the executable
    link_directories(/home/lizhifei/miniconda3/envs/trec2/lib)

    target_link_libraries(prof PRIVATE 
        ${TORCH_LIBRARIES} 
        pybind11::pybind11
        ${CUDA_LIBRARIES} 
        /home/lizhifei/miniconda3/envs/trec2/lib/libnvToolsExt.so 
        dl
        /home/lizhifei/miniconda3/envs/trec2/lib/libpython3.12.so
    )

    # Additional compiler options
    target_compile_options(prof PRIVATE -export-dynamic -fPIC -finstrument-functions)

else()
    # Create the executable
    add_executable(prof prof.cpp 
        /home/lizhifei/TREC-Artifact/build/temp.linux-x86_64-cpython-312/home/lizhifei/TREC-Artifact/trec_op/ConvDeepReuseBackward/convDR_backward.o
        /home/lizhifei/TREC-Artifact/build/temp.linux-x86_64-cpython-312/home/lizhifei/TREC-Artifact/trec_op/ConvDeepReuseForward/convDR_forward.o
    )

    # Link libraries to the executable
    link_directories(/home/lizhifei/miniconda3/envs/trec2/lib)

    target_link_libraries(prof PRIVATE 
        ${TORCH_LIBRARIES} 
        pybind11::pybind11
        ${CUDA_LIBRARIES} 
        /home/lizhifei/miniconda3/envs/trec2/lib/libpython3.12.so
    )

endif()