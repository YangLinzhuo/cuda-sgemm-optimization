//
// Created by linn on 9/29/23.
//

#ifndef SGEMM_KERNELS_CUH
#define SGEMM_KERNELS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))


void test_cublas(cublasHandle_t handle, int M, int N, int K,
                 float alpha, float *A, float *B, float beta, float *C);
void test_naive_kernel(cublasHandle_t handle, int M, int N, int K,
                       float alpha, float *A, float *B, float beta, float *C);
void test_sm_kernel(cublasHandle_t handle, int M, int N, int K,
                    float alpha, float *A, float *B, float beta, float *C);
void test_tile_1d_kernel(cublasHandle_t handle, int M, int N, int K,
                         float alpha, float *A, float *B, float beta, float *C);
void test_tile_2d_kernel(cublasHandle_t handle, int M, int N, int K,
                         float alpha, float *A, float *B, float beta, float *C);
void test_tile_2d_reg_cache_kernel(cublasHandle_t handle, int M, int N, int K,
                                   float alpha, float *A, float *B, float beta, float *C);
void test_tile_2d_float4_kernel(cublasHandle_t handle, int M, int N, int K,
                                float alpha, float *A, float *B, float beta, float *C);
void test_tile_2d_float4_double_buffering_kernel(cublasHandle_t handle, int M, int N, int K,
                                                 float alpha, float *A, float *B, float beta, float *C);
void test_no_share_conflict_kernel(cublasHandle_t handle, int M, int N, int K,
                                   float alpha, float *A, float *B, float beta, float *C);
void test_tile_2d_split_kernel(cublasHandle_t handle, int M, int N, int K,
                               float alpha, float *A, float *B, float beta, float *C);
void test_tile_1d_split_kernel(cublasHandle_t handle, int M, int N, int K,
                               float alpha, float *A, float *B, float beta, float *C);
#endif //SGEMM_KERNELS_CUH
