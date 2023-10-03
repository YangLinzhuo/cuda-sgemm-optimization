//
// Created by linn on 9/29/23.
//

#include "kernels.cuh"
#include <cstdio>

// A: (M, K)
// B: (K, N)
// C: (M, N)
void test_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    //cublas列主序计算：https://www.cnblogs.com/cuancuancuanhao/p/7763256.html
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}


#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))

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


template <const int BM, const int BN, const int BK>
__global__ void sm_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float val = 0.;
    int num_shared_block = CEIL_DIV(K, BK); // or CEIL_DIV(K, BN);
    A = &A[OFFSET(blockIdx.y * BM, 0, K)];
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    for (int i = 0; i < num_shared_block; ++i) {
        // Copy data from global memory to shared memory
        int A_row = threadIdx.y;
        int A_col = threadIdx.x;
        if ((blockIdx.y * BM + A_row) < M && (i * BK + A_col) < K) {
            As[threadIdx.y][threadIdx.x] = A[OFFSET(A_row, A_col, K)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.;
        }
        int B_row = threadIdx.y;
        int B_col = threadIdx.x;
        if ((i * BK + B_row) < K && (blockIdx.x * BN + B_col) < N) {
            Bs[threadIdx.y][threadIdx.x] = B[OFFSET(B_row, B_col, N)];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.;
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int k = 0; k < BK; ++k) {
            val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    int C_row = threadIdx.y;
    int C_col = threadIdx.x;
    if ((blockIdx.y * BM + C_row) < M && (blockIdx.x * BN + C_col) < N) {
        C[OFFSET(C_row, C_col, N)] = alpha * val + beta * C[OFFSET(C_row, C_col, N)];
    }
}


void test_sm_kernel(cublasHandle_t handle, int M, int N, int K,
                    float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, size), CEIL_DIV(M, size)); // note: change M and N here
    sm_kernel<size, size, size><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


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
    dim3 grid(CEIL_DIV(N, size * tile_size), CEIL_DIV(M, size * tile_size)); // note: change M and N here
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
    dim3 grid(CEIL_DIV(N, size * tile_size), CEIL_DIV(M, size * tile_size)); // note: change M and N here
    tile_2d_reg_cache_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void tile_2d_float4_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;
    int num_shared_block = CEIL_DIV(K, BK);

    __shared__ float As[BK][BM];    // transpose shared A for avoid bank conflict
    __shared__ float Bs[BK][BN];

    float accum[TM][TN] = {0.};

    const int load_a_cache_time = (BK * BM) / thread_num / 4;  // Each thread load 4 float
    const int load_b_cache_time = (BK * BN) / thread_num / 4;  // Each thread load 4 float

    float load_a_cache[4 * load_a_cache_time];

    A = &A[OFFSET(blockIdx.y * BM, 0, K)]; // Set block start position
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int a_tile_row = thread_id / (BK / 4);
    int a_tile_col = thread_id % (BK / 4) * 4;
    int a_tile_stride = BM / load_a_cache_time;
//    printf("A tile row, col, stride %d, %d, %d", a_tile_row, a_tile_col, a_tile_stride);

    int b_tile_row = thread_id / (BN / 4);
    int b_tile_col = thread_id % (BN / 4) * 4;
    int b_tile_stride = BK / load_b_cache_time;

    float As_cache[TM] = {0.};
    float Bs_cache[TN] = {0.};

#pragma unroll
    for (int i = 0; i < num_shared_block; ++i) {
#pragma unroll
        for (int m = 0; m < BM; m += a_tile_stride) {
            int cache_idx = m / a_tile_stride * 4;
            FETCH_FLOAT4(load_a_cache[cache_idx]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + m, a_tile_col, K)]);
            // Use load_a_cache for load 4 float at a time
            // As is saved as transpose matrix
            As[a_tile_col][a_tile_row + m] = load_a_cache[cache_idx];
            As[a_tile_col + 1][a_tile_row + m] = load_a_cache[cache_idx + 1];
            As[a_tile_col + 2][a_tile_row + m] = load_a_cache[cache_idx + 2];
            As[a_tile_col + 3][a_tile_row + m] = load_a_cache[cache_idx + 3];
        }
#pragma unroll
        for (int k = 0; k < BK; k += b_tile_stride) {
            FETCH_FLOAT4(Bs[b_tile_row + k][b_tile_col]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + k, b_tile_col, N)]);
        }
        __syncthreads();
        A += BK;    // Start position of next tile block to be processed
        B += BK * N;    // Start position of next tile block to be processed

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                int A_row = threadIdx.y * TM + m;
                FETCH_FLOAT4(As_cache[m]) = FETCH_FLOAT4(As[k][A_row]);
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                int B_col = threadIdx.x * TN + n;
                FETCH_FLOAT4(Bs_cache[n]) = FETCH_FLOAT4(Bs[k][B_col]);
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += As_cache[m] * Bs_cache[n];
                }
            }
        }
        __syncthreads();
    }

    float tmp[4] = {0.};
#pragma unroll
    for (int m = 0; m < TM; ++m) {
        int C_row = threadIdx.y * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int C_col = threadIdx.x * TN + n;
            FETCH_FLOAT4(tmp) = FETCH_FLOAT4(C[OFFSET(C_row, C_col, N)]);
            tmp[0] = alpha * accum[m][n] + beta * tmp[0];
            tmp[1] = alpha * accum[m][n + 1] + beta * tmp[1];
            tmp[2] = alpha * accum[m][n + 2] + beta * tmp[2];
            tmp[3] = alpha * accum[m][n + 3] + beta * tmp[3];
            FETCH_FLOAT4(C[OFFSET(C_row, C_col, N)]) = FETCH_FLOAT4(tmp);
        }
    }
}

void test_tile_2d_float4_kernel(cublasHandle_t handle, int M, int N, int K,
                                   float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
    const int tile_size = 8;
    const int BM = size * tile_size;
    const int BN = size * tile_size;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BM), CEIL_DIV(M, BN)); // note: change M and N here
    tile_2d_float4_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void tile_2d_float4_double_buffering_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;
    int num_shared_block = CEIL_DIV(K, BK);

    __shared__ float As[2][BK][BM];    // transpose shared A for avoid bank conflict, for double buffering
    __shared__ float Bs[2][BK][BN];    // for double buffering

    float accum[TM][TN] = {0.};

    const int load_a_cache_time = (BK * BM) / thread_num / 4;  // Each thread load 4 float
    const int load_b_cache_time = (BK * BN) / thread_num / 4;  // Each thread load 4 float

    float load_a_cache[4 * load_a_cache_time];
//    float load_a_cache[4];
//    float load_b_cache[4 * load_b_cache_time];

    A = &A[OFFSET(blockIdx.y * BM, 0, K)]; // Set block start position
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int a_tile_row = thread_id / (BK / 4);
    int a_tile_col = thread_id % (BK / 4) * 4;
    int a_tile_stride = BM / load_a_cache_time;
//    printf("A tile row, col, stride %d, %d, %d", a_tile_row, a_tile_col, a_tile_stride);

    int b_tile_row = thread_id / (BN / 4);
    int b_tile_col = thread_id % (BN / 4) * 4;
    int b_tile_stride = BK / load_b_cache_time;

    float As_cache[2][TM] = {0.};  // double buffering
    float Bs_cache[2][TN] = {0.};  // double buffering

    int write_idx = 0;

#pragma unroll
    for (int i = 0; i < num_shared_block; ++i) {
#pragma unroll
        for (int m = 0; m < BM; m += a_tile_stride) {
            int cache_idx = m / a_tile_stride * 4;
            FETCH_FLOAT4(load_a_cache[cache_idx]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + m, a_tile_col, K)]);
            // Use load_a_cache for load 4 float at a time
            // As is saved as transpose matrix
            As[write_idx][a_tile_col][a_tile_row + m] = load_a_cache[cache_idx];
            // 这里 stride = 128，有 shared memory bank 冲突
            As[write_idx][a_tile_col + 1][a_tile_row + m] = load_a_cache[cache_idx + 1];
            As[write_idx][a_tile_col + 2][a_tile_row + m] = load_a_cache[cache_idx + 2];
            As[write_idx][a_tile_col + 3][a_tile_row + m] = load_a_cache[cache_idx + 3];
        }
#pragma unroll
        for (int k = 0; k < BK; k += b_tile_stride) {
            FETCH_FLOAT4(Bs[write_idx][b_tile_row + k][b_tile_col]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + k, b_tile_col, N)]);
        }
        __syncthreads();
        A += BK;    // Start position of next tile block to be processed
        B += BK * N;    // Start position of next tile block to be processed

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                int A_row = threadIdx.y * TM + m;
                FETCH_FLOAT4(As_cache[write_idx][m]) = FETCH_FLOAT4(As[write_idx][k][A_row]);
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                int B_col = threadIdx.x * TN + n;
                FETCH_FLOAT4(Bs_cache[write_idx][n]) = FETCH_FLOAT4(Bs[write_idx][k][B_col]);
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += As_cache[write_idx][m] * Bs_cache[write_idx][n];
                }
            }
        }
        write_idx ^= 1;
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
        int C_row = threadIdx.y * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int C_col = threadIdx.x * TN + n;
            FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(C_row, C_col, N)]);
            load_a_cache[0] = alpha * accum[m][n] + beta * load_a_cache[0];
            load_a_cache[1] = alpha * accum[m][n + 1] + beta * load_a_cache[1];
            load_a_cache[2] = alpha * accum[m][n + 2] + beta * load_a_cache[2];
            load_a_cache[3] = alpha * accum[m][n + 3] + beta * load_a_cache[3];
            FETCH_FLOAT4(C[OFFSET(C_row, C_col, N)]) = FETCH_FLOAT4(load_a_cache);
        }
    }
}

void test_tile_2d_float4_double_buffering_kernel(cublasHandle_t handle, int M, int N, int K,
                                float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
    const int tile_size = 8;
    const int BM = size * tile_size;
    const int BN = size * tile_size;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BM), CEIL_DIV(M, BN)); // note: change M and N here
    tile_2d_float4_double_buffering_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


// solve shared memory bank conflict
// https://blog.csdn.net/Bruce_0712/article/details/65447608
// https://blog.csdn.net/sunmc1204953974/article/details/51078818
// x: warp 在执行时以 half-warp 为单位执行，分属于不同 warp 的线程之间不会有冲突
// 执行和调度以warp为单位，存储器访问以half-warp为单位。

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void no_share_conflict_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;
    const int THREAD_TILE = TM / 4;
    // The left and uppermost element position of thread tile in block
    int start_col = blockIdx.x * BN;
    int start_row = blockIdx.y * BM;
    int tx = threadIdx.x * TN;
    int ty = threadIdx.y * TM;

    __shared__ float As[2][BK][BM];    // transpose shared A for avoid bank conflict, for double buffering
    __shared__ float Bs[2][BK][BN];    // for double buffering

    float accum[TM][TN] = {0.};

    const int load_a_cache_time = (BK * BM) / thread_num / 4;  // Each thread load 4 float
    const int load_b_cache_time = (BK * BN) / thread_num / 4;  // Each thread load 4 float

//    float load_a_cache[4 * load_a_cache_time];
    float load_a_cache[4];
//    float load_b_cache[4 * load_b_cache_time];

    A = &A[OFFSET(start_row, 0, K)]; // Set block start position
    B = &B[OFFSET(0, start_col, N)];
    C = &C[OFFSET(start_row, start_col, N)];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int a_tile_row = thread_id / (BK / 4);
    int a_tile_col = thread_id % (BK / 4) * 4;
    int a_tile_stride = BM / load_a_cache_time;
//    printf("A tile row, col, stride %d, %d, %d", a_tile_row, a_tile_col, a_tile_stride);

    int b_tile_row = thread_id / (BN / 4);
    int b_tile_col = thread_id % (BN / 4) * 4;
    int b_tile_stride = BK / load_b_cache_time;

    float a_reg[2][TM] = {0.};  // double buffering
    float b_reg[2][TN] = {0.};  // double buffering

    int write_idx = 0;

#pragma unroll
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int cache_idx = i / a_tile_stride * 4;
            FETCH_FLOAT4(load_a_cache) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            // Use load_a_cache for load 4 float at a time
            // As is saved as transpose matrix
            As[write_idx][a_tile_col][a_tile_row + i] = load_a_cache[cache_idx];
            // 这里 stride = 128，有 shared memory bank 冲突
            As[write_idx][a_tile_col + 1][a_tile_row + i] = load_a_cache[cache_idx + 1];
            As[write_idx][a_tile_col + 2][a_tile_row + i] = load_a_cache[cache_idx + 2];
            As[write_idx][a_tile_col + 3][a_tile_row + i] = load_a_cache[cache_idx + 3];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(Bs[write_idx][b_tile_row + i][b_tile_col]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
        }
        __syncthreads();
        A += BK;    // Start position of next tile block to be processed
        B += BK * N;    // Start position of next tile block to be processed

#pragma unroll
        for (int i = 0; i < BK; ++i) {
#pragma unroll
            for (int t = 0; t < THREAD_TILE; ++t) {
                FETCH_FLOAT4(a_reg[write_idx][4 * t]) =
                        FETCH_FLOAT4(As[write_idx][i][ty / THREAD_TILE + t * BM / THREAD_TILE]);
            }
#pragma unroll
            for (int t = 0; t < THREAD_TILE; ++t) {
                FETCH_FLOAT4(b_reg[write_idx][t * 4]) =
                        FETCH_FLOAT4(Bs[write_idx][i][tx / THREAD_TILE + t * BM / THREAD_TILE]);
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += a_reg[write_idx][m] * b_reg[write_idx][n];
                }
            }
        }
        write_idx ^= 1;
    }

#pragma unroll
    for (int m = 0; m < TM / 2; ++m) {
        FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(ty / 2 + m, tx / 2, N)]);
        load_a_cache[0] = alpha * accum[m][0] + beta * load_a_cache[0];
        load_a_cache[1] = alpha * accum[m][1] + beta * load_a_cache[1];
        load_a_cache[2] = alpha * accum[m][2] + beta * load_a_cache[2];
        load_a_cache[3] = alpha * accum[m][3] + beta * load_a_cache[3];
        FETCH_FLOAT4(C[OFFSET(ty / 2 + m, tx / 2, N)]) = FETCH_FLOAT4(load_a_cache);
        FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(ty / 2 + m, tx / 2, N) + BN / 2]);
        load_a_cache[0] = alpha * accum[m][4] + beta * load_a_cache[0];
        load_a_cache[1] = alpha * accum[m][5] + beta * load_a_cache[1];
        load_a_cache[2] = alpha * accum[m][6] + beta * load_a_cache[2];
        load_a_cache[3] = alpha * accum[m][7] + beta * load_a_cache[3];
        FETCH_FLOAT4(C[OFFSET(ty / 2 + m, tx / 2, N) + BN / 2]) = FETCH_FLOAT4(load_a_cache);
    }

#pragma unroll
    for (int m = 0; m < TM / 2; ++m) {
        FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(ty / 2 + m + BN / 2, tx / 2, N)]);
        load_a_cache[0] = alpha * accum[m + TM / 2][0] + beta * load_a_cache[0];
        load_a_cache[1] = alpha * accum[m + TM / 2][1] + beta * load_a_cache[1];
        load_a_cache[2] = alpha * accum[m + TM / 2][2] + beta * load_a_cache[2];
        load_a_cache[3] = alpha * accum[m + TM / 2][3] + beta * load_a_cache[3];
        FETCH_FLOAT4(C[OFFSET(ty / 2 + m + BN / 2, tx / 2, N)]) = FETCH_FLOAT4(load_a_cache);
        FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(ty / 2 + m + BN / 2, tx / 2, N) + BN / 2]);
        load_a_cache[0] = alpha * accum[m + TM / 2][4] + beta * load_a_cache[0];
        load_a_cache[1] = alpha * accum[m + TM / 2][5] + beta * load_a_cache[1];
        load_a_cache[2] = alpha * accum[m + TM / 2][6] + beta * load_a_cache[2];
        load_a_cache[3] = alpha * accum[m + TM / 2][7] + beta * load_a_cache[3];
        FETCH_FLOAT4(C[OFFSET(ty / 2 + m + BN / 2, tx / 2, N) + BN / 2]) = FETCH_FLOAT4(load_a_cache);
    }
}


void test_no_share_conflict_kernel(cublasHandle_t handle, int M, int N, int K,
                                                 float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
    const int tile_size = 8;
    const int BM = size * tile_size;
    const int BN = size * tile_size;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BM), CEIL_DIV(M, BN)); // note: change M and N here
    no_share_conflict_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void tile_2d_split_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;
    int num_shared_block = CEIL_DIV(K, BK);

    __shared__ float As[2][BK][BM];    // transpose shared A for avoid bank conflict, for double buffering
    __shared__ float Bs[2][BK][BN];    // for double buffering

    float accum[TM][TN] = {0.};

    const int load_a_cache_time = (BK * BM) / thread_num / 4;  // Each thread load 4 float
    const int load_b_cache_time = (BK * BN) / thread_num / 4;  // Each thread load 4 float

    float load_a_cache[4 * load_a_cache_time];
//    float load_a_cache[4];
//    float load_b_cache[4 * load_b_cache_time];

    A = &A[OFFSET(blockIdx.y * BM, 0, K)]; // Set block start position
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int a_tile_row = thread_id / (BK / 4);
    int a_tile_col = thread_id % (BK / 4) * 4;
    int a_tile_stride = BM / load_a_cache_time;
//    printf("A tile row, col, stride %d, %d, %d", a_tile_row, a_tile_col, a_tile_stride);

    int b_tile_row = thread_id / (BN / 4);
    int b_tile_col = thread_id % (BN / 4) * 4;
    int b_tile_stride = BK / load_b_cache_time;

    float As_cache[2][TM] = {0.};  // double buffering
    float Bs_cache[2][TN] = {0.};  // double buffering

    int write_idx = 0;

#pragma unroll
    for (int i = 0; i < num_shared_block; ++i) {
#pragma unroll
        for (int m = 0; m < BM; m += a_tile_stride) {
            int cache_idx = m / a_tile_stride * 4;
            FETCH_FLOAT4(load_a_cache[cache_idx]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + m, a_tile_col, K)]);
            // Use load_a_cache for load 4 float at a time
            // As is saved as transpose matrix
            As[write_idx][a_tile_col][a_tile_row + m] = load_a_cache[cache_idx];
            // 这里 stride = 128，有 shared memory bank 冲突
            As[write_idx][a_tile_col + 1][a_tile_row + m] = load_a_cache[cache_idx + 1];
            As[write_idx][a_tile_col + 2][a_tile_row + m] = load_a_cache[cache_idx + 2];
            As[write_idx][a_tile_col + 3][a_tile_row + m] = load_a_cache[cache_idx + 3];
        }
#pragma unroll
        for (int k = 0; k < BK; k += b_tile_stride) {
            FETCH_FLOAT4(Bs[write_idx][b_tile_row + k][b_tile_col]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + k, b_tile_col, N)]);
        }
        __syncthreads();
        A += BK;    // Start position of next tile block to be processed
        B += BK * N;    // Start position of next tile block to be processed

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
//            for (int m = 0; m < TM; m += 4) {
            for (int m = 0, mm = 0; m < BM && mm < TM; m += block_row_thread * 4, mm += 4) {
//                int A_row = threadIdx.y * TM + m;
                int A_row = m + threadIdx.y * 4;
                FETCH_FLOAT4(As_cache[write_idx][mm]) = FETCH_FLOAT4(As[write_idx][k][A_row]);
            }
#pragma unroll
//            for (int n = 0; n < TN; n += 4) {
            for (int n = 0, nn = 0; n < BN && nn < TN; n += block_col_thread * 4, nn += 4) {
//                int B_col = threadIdx.x * TN + n;
                int B_col = n + threadIdx.x * 4;
                FETCH_FLOAT4(Bs_cache[write_idx][nn]) = FETCH_FLOAT4(Bs[write_idx][k][B_col]);
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += As_cache[write_idx][m] * Bs_cache[write_idx][n];
                }
            }
        }
        write_idx ^= 1;
    }

#pragma unroll
    for (int m = 0; m < TM; m += 4) {
        int C_row = (m / 4) * (block_row_thread * 4) + threadIdx.y * 4;
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int C_col = (n / 4) * (block_col_thread * 4) + threadIdx.x * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(C_row + i, C_col, N)]);
                load_a_cache[0] = alpha * accum[m + i][n] + beta * load_a_cache[0];
                load_a_cache[1] = alpha * accum[m + i][n + 1] + beta * load_a_cache[1];
                load_a_cache[2] = alpha * accum[m + i][n + 2] + beta * load_a_cache[2];
                load_a_cache[3] = alpha * accum[m + i][n + 3] + beta * load_a_cache[3];
                FETCH_FLOAT4(C[OFFSET(C_row + i, C_col, N)]) = FETCH_FLOAT4(load_a_cache);
            }
        }
    }
}

void test_tile_2d_split_kernel(cublasHandle_t handle, int M, int N, int K,
                               float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
//    const int size = 2;
    const int tile_size = 8;
    const int BM = size * tile_size;
    const int BN = size * tile_size;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BM), CEIL_DIV(M, BN)); // note: change M and N here
    tile_2d_split_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void tile_1d_split_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;
    int num_shared_block = CEIL_DIV(K, BK);

    __shared__ float As[2][BK][BM];    // transpose shared A for avoid bank conflict, for double buffering
    __shared__ float Bs[2][BK][BN];    // for double buffering

    float accum[TM][TN] = {0.};

    const int load_a_cache_time = (BK * BM) / thread_num / 4;  // Each thread load 4 float
    const int load_b_cache_time = (BK * BN) / thread_num / 4;  // Each thread load 4 float

    float load_a_cache[4 * load_a_cache_time];
//    float load_a_cache[4];
//    float load_b_cache[4 * load_b_cache_time];

    A = &A[OFFSET(blockIdx.y * BM, 0, K)]; // Set block start position
    B = &B[OFFSET(0, blockIdx.x * BN, N)];
    C = &C[OFFSET(blockIdx.y * BM, blockIdx.x * BN, N)];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int a_tile_row = thread_id / (BK / 4);
    int a_tile_col = thread_id % (BK / 4) * 4;
    int a_tile_stride = BM / load_a_cache_time;
//    printf("A tile row, col, stride %d, %d, %d", a_tile_row, a_tile_col, a_tile_stride);

    int b_tile_row = thread_id / (BN / 4);
    int b_tile_col = thread_id % (BN / 4) * 4;
    int b_tile_stride = BK / load_b_cache_time;

    float As_cache[2][TM] = {0.};  // double buffering
    float Bs_cache[2][TN] = {0.};  // double buffering

    int write_idx = 0;

#pragma unroll
    for (int i = 0; i < num_shared_block; ++i) {
#pragma unroll
        for (int m = 0; m < BM; m += a_tile_stride) {
            int cache_idx = m / a_tile_stride * 4;
            FETCH_FLOAT4(load_a_cache[cache_idx]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + m, a_tile_col, K)]);
            // Use load_a_cache for load 4 float at a time
            // As is saved as transpose matrix
            As[write_idx][a_tile_col][a_tile_row + m] = load_a_cache[cache_idx];
            // 这里 stride = 128，有 shared memory bank 冲突
            As[write_idx][a_tile_col + 1][a_tile_row + m] = load_a_cache[cache_idx + 1];
            As[write_idx][a_tile_col + 2][a_tile_row + m] = load_a_cache[cache_idx + 2];
            As[write_idx][a_tile_col + 3][a_tile_row + m] = load_a_cache[cache_idx + 3];
        }
#pragma unroll
        for (int k = 0; k < BK; k += b_tile_stride) {
            FETCH_FLOAT4(Bs[write_idx][b_tile_row + k][b_tile_col]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + k, b_tile_col, N)]);
        }
        __syncthreads();
        A += BK;    // Start position of next tile block to be processed
        B += BK * N;    // Start position of next tile block to be processed

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
//            for (int m = 0; m < TM; m += 4) {
            for (int m = 0, mm = 0; m < BM && mm < TM; m += block_row_thread * 4, mm += 4) {
//                int A_row = threadIdx.y * TM + m;
                int A_row = m + threadIdx.y * 4;
                FETCH_FLOAT4(As_cache[write_idx][mm]) = FETCH_FLOAT4(As[write_idx][k][A_row]);
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
//            for (int n = 0, nn = 0; n < BN && nn < TN; n += block_col_thread * 4, nn += 4) {
                int B_col = threadIdx.x * TN + n;
//                int B_col = n + threadIdx.x * 4;
                FETCH_FLOAT4(Bs_cache[write_idx][n]) = FETCH_FLOAT4(Bs[write_idx][k][B_col]);
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    accum[m][n] += As_cache[write_idx][m] * Bs_cache[write_idx][n];
                }
            }
        }
        write_idx ^= 1;
    }

#pragma unroll
    for (int m = 0; m < TM; m += 4) {
        int ROW = m / 4;
        int C_row = ROW * (block_row_thread * 4) + threadIdx.y * 4;
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int C_col = threadIdx.x * TN + n;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                FETCH_FLOAT4(load_a_cache) = FETCH_FLOAT4(C[OFFSET(C_row + i, C_col, N)]);
                load_a_cache[0] = alpha * accum[m + i][n] + beta * load_a_cache[0];
                load_a_cache[1] = alpha * accum[m + i][n + 1] + beta * load_a_cache[1];
                load_a_cache[2] = alpha * accum[m + i][n + 2] + beta * load_a_cache[2];
                load_a_cache[3] = alpha * accum[m + i][n + 3] + beta * load_a_cache[3];
                FETCH_FLOAT4(C[OFFSET(C_row + i, C_col, N)]) = FETCH_FLOAT4(load_a_cache);
            }
        }
    }
}

void test_tile_1d_split_kernel(cublasHandle_t handle, int M, int N, int K,
                               float alpha, float *A, float *B, float beta, float *C) {
    const int size = 16;
//    const int size = 2;
    const int tile_size = 8;
    const int BM = size * tile_size;
    const int BN = size * tile_size;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 block(size, size);
    dim3 grid(CEIL_DIV(N, BM), CEIL_DIV(M, BN)); // note: change M and N here
    tile_1d_split_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
