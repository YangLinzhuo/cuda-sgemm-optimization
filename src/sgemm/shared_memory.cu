//
// Created by YangLinzhuo on 2023/10/12.
//

#include "kernels.cuh"

template <const int BM, const int BN, const int BK>
__global__ void sm_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float val = 0.;

    for (int i = 0; i < CEIL_DIV(K, BK); ++i) {
        // Copy data from global memory to shared memory
        int A_row = blockIdx.y * BM + threadIdx.y;
        int A_col = i * BK + threadIdx.x;
        if (A_row < M && A_col < K) {
            As[threadIdx.y][threadIdx.x] = A[OFFSET(A_row, A_col, K)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.;
        }
        int B_row = i * BK + threadIdx.y;
        int B_col = blockIdx.x * BN + threadIdx.x;
        if (B_row < K && B_col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[OFFSET(B_row, B_col, N)];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    int C_row = blockIdx.y * BM + threadIdx.y;
    int C_col = blockIdx.x * BN + threadIdx.x;
    if (C_row < M && C_col < N) {
        C[OFFSET(C_row, C_col, N)] = alpha * val + beta * C[OFFSET(C_row, C_col, N)];
    }
}

template <const int BM, const int BN, const int BK>
__global__ void sm_transposed_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float val = 0.;

    for (int i = 0; i < CEIL_DIV(K, BK); ++i) {
        // Copy data from global memory to shared memory
        int A_row = blockIdx.y * BM + threadIdx.y;
        int A_col = i * BK + threadIdx.x;
        if (A_row < M && A_col < K) {
            As[threadIdx.y][threadIdx.x] = A[OFFSET(A_row, A_col, K)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.;
        }
        int B_row = i * BK + threadIdx.y;
        int B_col = blockIdx.x * BN + threadIdx.x;
        if (B_row < K && B_col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[OFFSET(B_row, B_col, N)];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    int C_row = blockIdx.y * BM + threadIdx.y;
    int C_col = blockIdx.x * BN + threadIdx.x;
    if (C_row < M && C_col < N) {
        C[OFFSET(C_row, C_col, N)] = alpha * val + beta * C[OFFSET(C_row, C_col, N)];
    }
}


void test_sm_kernel(cublasHandle_t handle, int M, int N, int K,
                    float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, size), CEIL_DIV(M, size)); // note: change M and N here
    sm_kernel<size, size, size><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
//    sm_transposed_kernel<size, size, size><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}