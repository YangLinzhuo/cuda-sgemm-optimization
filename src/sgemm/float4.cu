//
// Created by linn on 10/14/23.
//

#include "sgemm.cuh"

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
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // note: change M and N here
    tile_2d_float4_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

