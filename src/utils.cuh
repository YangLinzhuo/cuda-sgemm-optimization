//
// Created by linn on 9/29/23.
//

#ifndef SGEMM_UTILS_CUH
#define SGEMM_UTILS_CUH

#include <ctime>
#include <cstdio>
//#include <sys/time.h>
#include <cuda_runtime.h>

// https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
void checkLastError(const char* const file, const int line);
void checkError(cudaError_t err, const char* const func, const char* const file, const int line);
bool verify_matrix(const float *mat1, const float *mat2, int N);
void copy_matrix(const float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N);
void randomize_matrix(float *mat, int N);
void CudaDeviceInfo();

#define CHECK_CUDA_ERROR(val) checkError((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLastError(__FILE__, __LINE__)

#endif //SGEMM_UTILS_CUH
