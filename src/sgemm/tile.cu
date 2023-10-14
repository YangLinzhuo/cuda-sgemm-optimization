//
// Created by linn on 10/14/23.
//

#include "sgemm.cuh"

template <const int BM, const int BN, const int BK, const int TM>
__global__ void tile_1d_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float val[TM] = {0.};
    int num_shared_block = CEIL_DIV(K, BK); // or CEIL_DIV(K, BN);
    A = &A[OFFSET(blockIdx.y * BM, 0, K)];
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    for (int i = 0; i < num_shared_block; ++i) {
        // Copy data from global memory to shared memory
        for (int m = 0; m < TM; ++m) {
            int A_row = threadIdx.y * TM + m;
            int A_col = threadIdx.x;
            if ((blockIdx.y * BM + A_row) < M && (i * BK + A_col) < K) {
                As[A_row][A_col] = A[OFFSET(A_row, A_col, K)];
            } else {
                As[A_row][A_col] = 0.;
            }
        }
        int B_row = threadIdx.y;
        int B_col = threadIdx.x;
        if ((i * BK + B_row) < K && (blockIdx.x * BN + B_col) < N) {
            Bs[B_row][B_col] = B[OFFSET(B_row, B_col, N)];
        } else {
            Bs[B_row][B_col] = 0.;
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m) {
                int A_row = threadIdx.y * TM + m;
                int B_col = threadIdx.x;
                val[m] += As[A_row][k] * Bs[k][B_col];
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        int C_row = threadIdx.y * TM + m;
        int C_col = threadIdx.x;
        if ((blockIdx.y * BM + C_row) < M && (blockIdx.x * BN + C_col) < N) {
            C[OFFSET(C_row, C_col, N)] = alpha * val[m] + beta * C[OFFSET(C_row, C_col, N)];
        }
    }
}



void test_tile_1d_kernel(cublasHandle_t handle, int M, int N, int K,
                         float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
    const int tile_size = 8;
    const int BM = size * tile_size;
    const int BN = size;
    const int BK = size;
    const int TM = tile_size;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // note: change M and N here
    tile_1d_kernel<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void tile_2d_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float val[TM][TN] = {0.};
    int num_shared_block = CEIL_DIV(K, BK); // or CEIL_DIV(K, BN);
    A = &A[OFFSET(blockIdx.y * BM, 0, K)];
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    for (int i = 0; i < num_shared_block; ++i) {
        // Copy data from global memory to shared memory
        for (int m = 0; m < TM; ++m) {
            int A_row = threadIdx.y * TM + m;
            int A_col = threadIdx.x;
            if ((blockIdx.y * BM + A_row) < M && (i * BK + A_col) < K) {
                As[A_row][A_col] = A[OFFSET(A_row, A_col, K)];
            } else {
                As[A_row][A_col] = 0.;
            }
        }
        for (int n = 0; n < TN; ++n) {
            int B_row = threadIdx.y;
            int B_col = threadIdx.x * TN + n;
            if ((i * BK + B_row) < K && (blockIdx.x * BN + B_col) < N) {
                Bs[B_row][B_col] = B[OFFSET(B_row, B_col, N)];
            } else {
                Bs[B_row][B_col] = 0.;
            }
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m) {
                int A_row = threadIdx.y * TM + m;
                for (int n = 0; n < TN; ++n) {
                    int B_col = threadIdx.x * TN + n;
                    val[m][n] += As[A_row][k] * Bs[k][B_col];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        int C_row = threadIdx.y * TM + m;
        for (int n = 0; n < TN; ++n) {
            int C_col = threadIdx.x * TN + n;
            if ((blockIdx.y * BM + C_row) < M && (blockIdx.x * BN + C_col) < N) {
                C[OFFSET(C_row, C_col, N)] = alpha * val[m][n] + beta * C[OFFSET(C_row, C_col, N)];
            }
        }
    }
}

void test_tile_2d_kernel(cublasHandle_t handle, int M, int N, int K,
                         float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
    const int tile_size = 4;
    const int BM = size * tile_size;
    const int BN = size * tile_size;
    const int BK = size;
    const int TM = tile_size;
    const int TN = tile_size;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // note: change M and N here
    tile_2d_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void tile_2d_reg_cache_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float As_cache[TM] = {0.};
    float Bs_cache[TN] = {0.};
    float val[TM][TN] = {0.};

    int num_shared_block = CEIL_DIV(K, BK);
    A = &A[OFFSET(blockIdx.y * BM, 0, K)];
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    for (int i = 0; i < num_shared_block; ++i) {
        // Copy data from global memory to shared memory
        for (int m = 0; m < TM; ++m) {
            int A_row = threadIdx.y * TM + m;
            int A_col = threadIdx.x;
            if ((blockIdx.y * BM + A_row) < M && (i * BK + A_col) < K) {
                As[A_row][A_col] = A[OFFSET(A_row, A_col, K)];
            } else {
                As[A_row][A_col] = 0.;
            }
        }
        for (int n = 0; n < TN; ++n) {
            int B_row = threadIdx.y;
            int B_col = threadIdx.x * TN + n;
            if ((i * BK + B_row) < K && (blockIdx.x * BN + B_col) < N) {
                Bs[B_row][B_col] = B[OFFSET(B_row, B_col, N)];
            } else {
                Bs[B_row][B_col] = 0.;
            }
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m) {
                int A_row = threadIdx.y * TM + m;
                As_cache[m] = As[A_row][k];
            }
            for (int n = 0; n < TN; ++n) {
                int B_col = threadIdx.x * TN + n;
                Bs_cache[n] = Bs[k][B_col];
            }
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    val[m][n] += As_cache[m] * Bs_cache[n];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        int C_row = threadIdx.y * TM + m;
        for (int n = 0; n < TN; ++n) {
            int C_col = threadIdx.x * TN + n;
            if ((blockIdx.y * BM + C_row) < M && (blockIdx.x * BN + C_col) < N) {
                C[OFFSET(C_row, C_col, N)] = alpha * val[m][n] + beta * C[OFFSET(C_row, C_col, N)];
            }
        }
    }
}


void test_tile_2d_reg_cache_kernel(cublasHandle_t handle, int M, int N, int K,
                                   float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
    const int tile_size = 4;
    const int BM = size * tile_size;
    const int BN = size * tile_size;
    const int BK = size;
    const int TM = tile_size;
    const int TN = tile_size;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // note: change M and N here
    tile_2d_reg_cache_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
