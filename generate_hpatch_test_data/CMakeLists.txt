cmake_minimum_required(VERSION 3.10)
project(generate_hpatch_test_data)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# TensorRT include/lib
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include /usr/include/x86_64-linux-gnu /usr/local/include)
find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)

# CUDA runtime lib
find_library(CUDA_RUNTIME_LIBRARY cudart PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

add_executable(extract_features_trt
    extract_features_trt.cc
    ORBextractor.cc
)

target_link_libraries(extract_features_trt
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${CUDA_RUNTIME_LIBRARY}
    cnpy z
)
