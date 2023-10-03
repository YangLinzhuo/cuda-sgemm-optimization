//
// Created by linn on 9/29/23.
//

#include <iostream>
#include "src/utils.cuh"
#include "src/kernels.cuh"

int main(int argc, const char* argv[]) {
    if (argc == 1) {
        CudaDeviceInfo();
        return 0;
    }

    if (argc != 3 && argc != 5) {
        printf("Please select a kernel and corresponding matrix size.\n");
        printf("Max kernel size is 6400.\n");
        printf("Kernel 0 is for cuBLAS kernel implemented by NVIDIA.\n");
        std::exit(EXIT_FAILURE);
    }

    // 申明句柄，创建句柄, cublasCreate会返回一个cublasStatus_t类型的值，用来判断句柄是否创建成功(值为0)
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle)) {
        printf("Create cublas handle error.\n");
        std::exit(EXIT_FAILURE);
    }

    float elapsed_time = 0.;
    cudaEvent_t beg = nullptr, end = nullptr;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    int kernel = atoi(argv[1]);

    using func_ptr = void (*)(cublasHandle_t handle, int M, int N, int K,
            float alpha, float *A, float *B, float beta, float *C);
    func_ptr test_func = nullptr;
    switch (kernel) {
        case 1:
            test_func = test_naive_kernel;
            break;
        case 2:
            test_func = test_sm_kernel;
            break;
        case 3:
            test_func = test_tile_1d_kernel;
            break;
        case 4:
            test_func = test_tile_2d_kernel;
            break;
        case 5:
            test_func = test_tile_2d_reg_cache_kernel;
            break;
        case 6:
            test_func = test_tile_2d_float4_kernel;
            break;
        case 7:
            test_func = test_tile_2d_float4_double_buffering_kernel;
            break;
        case 8:
            test_func = test_no_share_conflict_kernel;
            break;
        case 9:
            test_func = test_tile_1d_split_kernel;
            break;
        case 10:
            test_func = test_tile_2d_split_kernel;
            break;
        default:
            test_func = test_cublas;
            break;
    }

    int m = 0, n = 0, k = 0;
    if (argc == 3) {
        m = n = k = atoi(argv[2]);
    } else if (argc == 5) {
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }

    float alpha = 1.0, beta = 0.; //two arbitrary input parameters，C=α*AB+β*C
    float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;     //host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; //device matrices

    A = new float[m * k];
    B = new float[k * n];
    C = new float[m * n];
    C_ref = new float[m * n];

    CHECK_CUDA_ERROR(cudaMalloc((void **) &dA, sizeof(float) * m * k));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &dB, sizeof(float) * k * n));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &dC, sizeof(float) * m * n));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &dC_ref, sizeof(float) * m * n));

    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);
    randomize_matrix(C, m * n);
    copy_matrix(C, C_ref, m * n);

    CHECK_CUDA_ERROR(cudaMemcpy(dA, A, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dB, B, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dC, C, sizeof(float) * m * n, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dC_ref, C_ref, sizeof(float) * m * n, cudaMemcpyHostToDevice));

//    test_cublas(handle, m, n, k, alpha, dA, dB, beta, dC);
    test_func(handle, m, n, k, alpha, dA, dB, beta, dC);
    test_cublas(handle, m, n, k, alpha, dA, dB, beta, dC_ref);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();

    if (!verify_matrix(C_ref, C, m * n)) {
        printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
        print_matrix(C, m, n);
        print_matrix(C_ref, m, n);
        std::exit(EXIT_FAILURE);
    }

    int repeat_times = 5;
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
        test_func(handle, m, n, k, alpha, dA, dB, beta, dC);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; //换算成秒

//    printf("Average elapsed time: (%f) second, performance: (%f) GFLOPS. size: (%d).\n",
//           elapsed_time / repeat_times, 2. * 1e-9 * repeat_times * m * n * k / elapsed_time, m);
    printf("%f %f %d\n",
           elapsed_time / repeat_times, 2. * 1e-9 * repeat_times * m * n * k / elapsed_time, m);

    // 释放CPU和GPU空间
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;
    CHECK_CUDA_ERROR(cudaFree(dA));
    CHECK_CUDA_ERROR(cudaFree(dB));
    CHECK_CUDA_ERROR(cudaFree(dC));
    CHECK_CUDA_ERROR(cudaFree(dC_ref));
    CHECK_LAST_CUDA_ERROR();
    return 0;
}