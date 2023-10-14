//
// Created by linn on 9/29/23.
//

#include "sgemm.cuh"
#include <cstdio>

// A: (M, K)
// B: (K, N)
// C: (M, N)
void test_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    //cublas列主序计算：https://www.cnblogs.com/cuancuancuanhao/p/7763256.html
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}


__global__ void naive_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float val = 0.;
        for (int k = 0; k < K; ++k) {
            val += A[OFFSET(row, k, K)] * B[OFFSET(k, col, N)];
        }
        C[OFFSET(row, col, N)] = alpha * val + beta * C[OFFSET(row, col, N)];
    }
}

void test_naive_kernel(cublasHandle_t handle, int M, int N, int K,
                       float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, size), CEIL_DIV(M, size)); // note: change M and N here
    naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
