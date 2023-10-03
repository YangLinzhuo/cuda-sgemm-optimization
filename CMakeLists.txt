cmake_minimum_required(VERSION 3.22)
project(sgemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "-O3")
#set(CMAKE_CUDA_FLAGS "-g -G")

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} main.cu
        src/utils.cu
        src/kernels.cu)

# 可执行文件输出路径
# https://gist.github.com/gavinb/c993f71cf33d2354515c4452a3f8ef30
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
# 查询 compute capability https://developer.nvidia.com/cuda-gpus
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "86")

# 配置头文件搜索路径
# 配置 CUDA 相关库头文件
# 参考
# https://stackoverflow.com/questions/51756562/obtaining-the-cuda-include-dir-in-c-targets-with-native-cuda-support-cmake
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

# link cudart cublas
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas)