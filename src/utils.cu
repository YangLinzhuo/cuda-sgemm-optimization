//
// Created by linn on 9/29/23.
//

#include <iostream>
#include <cstdio>
#include "utils.cuh"

void checkError(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << "[Error]" << cudaGetErrorString(err) << " when call " << func << std::endl;
         std::exit(EXIT_FAILURE);
    }
}

void checkLastError(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
         std::exit(EXIT_FAILURE);
    }
}


bool verify_matrix(const float *mat1, const float *mat2, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-1) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i], mat2[i], i);
            return false;
        }
    }
    return true;
}


void randomize_matrix(float *mat, int N) {
    auto const seed = 1101;
    std::mt19937 engine {seed};
    std::uniform_real_distribution<float> generator {-5.f, 5.f};
    for (int i = 0; i < N; i++) {
        mat[i] = generator(engine);
    }
}


void copy_matrix(const float *src, float *dest, int N) {
    int i;
    for (i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if (i != N)
        printf("copy failed at %d while there are %d elements in total.\n", i, N);
}


void print_matrix(const float *A, int M, int N) {
    printf("[\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%5.2f ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("]\n");
}


void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    /*
   * There should be no need to modify the output string below.
   */

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           deviceId,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
};