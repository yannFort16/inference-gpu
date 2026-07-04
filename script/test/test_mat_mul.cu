// Compile : nvcc -m64 -diag-suppress 2464  -o matrix_mult.exe .\script\test\tests_mat_mut.cu 

#include <stdio.h>
#include <stdlib.h>
#include "../header/mat_mul.h"
#include "../header/utils.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"


int cublas_gemm(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, bool perf) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, afterH2D, afterKernel, afterD2H;
    cudaEventCreate(&start);
    cudaEventCreate(&afterH2D);
    cudaEventCreate(&afterKernel);
    cudaEventCreate(&afterD2H);

    float *d_A;
    float *d_B;
    float *d_C;

    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);


    // Allocate device memory
    cudaMalloc((void**)&d_A, bytes_A);
    cudaMalloc((void**)&d_B, bytes_B);
    cudaMalloc((void**)&d_C, bytes_C);

    cudaEventRecord(start);
    cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, bytes_C, cudaMemcpyHostToDevice);
    cudaEventRecord(afterH2D);
    // Perform the matrix multiplication: C = alpha * A * B + beta * C
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                            n, m, k, &alpha, d_B, n, d_A, 
                                            k, &beta, d_C, n);
    cudaDeviceSynchronize();
    cudaEventRecord(afterKernel);
    
    cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost);
    cudaEventRecord(afterD2H);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (perf) {
        float h2d_ms, kernel_ms, d2h_ms, total_ms;
        cudaEventElapsedTime(&h2d_ms, start, afterH2D);
        cudaEventElapsedTime(&kernel_ms, afterH2D, afterKernel);
        cudaEventElapsedTime(&d2h_ms, afterKernel, afterD2H);
        cudaEventElapsedTime(&total_ms, start, afterD2H);
        print_performance(h2d_ms, kernel_ms, d2h_ms, total_ms, "cuBLAS");
    }

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed with error code %d\n", stat);
        return -1;
    }
    return 0;
}

int validate_matrix_multiplication() {
    printf("===== Test CUDA Matrix Multiplication =====\n\n");
    
    const int nb_tests = 4; 

    int test[nb_tests][3] = {
        {1080, 1920, 9},
        {64, 25, 49},
        {128, 128, 64},
        {256, 256, 1024}
    }; // Tests with {M, N, K}
    
    for(int t = 0; t<nb_tests; t++){
        int m = test[t][0];
        int n = test[t][1];
        int k = test[t][2];
        printf("Test with Matrix dimensions: A[%d x %d], B[%d x %d], C[%d x %d]\n", m, k, k, n, m, n);

        // Allocate host memory
        size_t bytes_A = m * k * sizeof(float);
        size_t bytes_B = k * n * sizeof(float);
        size_t bytes_C = m * n * sizeof(float);
        
        float *h_A = (float*)malloc(bytes_A);
        float *h_B = (float*)malloc(bytes_B);
        float *h_C1 = (float*)malloc(bytes_C);
        float *h_C2 = (float*)malloc(bytes_C);
        
        // Initialize C with zeros
        for (int i = 0; i < m * n; i++){
            h_C1[i] = 0.0;
            h_C2[i] = 0.0;
        } 

        generateRandomMatrix(h_A, m, k, 50, false);
        generateRandomMatrix(h_B, k, n, 50, false);
    
        int res = matrix_multiplication(h_A, h_B, h_C1, m, n, k, "cuBLAS", true);
        if (res != 0) {
            printf("Matrix Multiplication failed with return code %d\n", res);
            continue;
        }

        res = matrix_multiplication(h_A, h_B, h_C2, m, n, k, "default", true);
        if (res != 0) {
            printf("Matrix Multiplication failed with return code %d\n", res);
            continue;
        }

        bool v = compare_matrix(h_C1, h_C2, m, n);

        if (v) {
            printf("Test %d passed\n", t);
        } else {
            printf("Test %d FAILED\n", t);
        }
        printf("\n---------------------------\n");
        free(h_A);
        free(h_B);
        free(h_C1);
        free(h_C2);
    }
    printf("Done!\n");
    return 0;
}

int benchmark_matrix_multiplication(int m, int n, int k) {
    printf("===== Benchmark CUDA Matrix Multiplication =====\n\n");
    
    const int loop_count = 32; 

    char* methods[2] = {"cuBLAS", "sharedM"};

    printf("Benchmark with Matrix dimensions: A[%d x %d], B[%d x %d], C[%d x %d]\n", m, k, k, n, m, n);


    // Allocate host memory
    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);
        
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C1 = (float*)malloc(bytes_C);
    float *h_C2 = (float*)malloc(bytes_C);

    
    bool last_iter = false;
    //Create Handle:
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed (code %d)\n", stat);
        return EXIT_FAILURE;
    }
    
    for(int t = 0; t<loop_count; t++){    
        generateRandomMatrix(h_A, m, k, 50, false);
        generateRandomMatrix(h_B, k, n, 50, false);
        
        last_iter = (t == loop_count - 1);
            
        int res = cublas_gemm(handle, h_A, h_B, h_C1, m, n, k, last_iter);
        if (res != 0) {
            printf("Matrix Multiplication %s failed with return code %d\n", methods[0], res);
            continue;
        }

        res = matrix_multiplication(h_A, h_B, h_C2, m, n, k, methods[1], last_iter);
        if (res != 0) {
            printf("Matrix Multiplication %s failed with return code %d\n", methods[1], res);
            continue;
        }

        if (last_iter) {
            compare_matrix(h_C1, h_C2, m, n); 
        }
        //printf("%d/%d iterations completed\n", t+1, loop_count);
    }

    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);
    cublasDestroy(handle);
    printf("Done!\n");
    return 0;
}

int main() {
    //validate_matrix_multiplication();

    int m = 256;
    int n = 3136;
    int k = 2304;
    
    const int nb_tests = 4;

    int test[nb_tests][3] = {
        {64, 50176, 27},
        {256, 3136, 2304},
        {512, 784, 4608},
        {128, 12544, 1152}
    };
    for(int t = 0; t<nb_tests; t++){
        m = test[t][0];
        n = test[t][1];
        k = test[t][2];
        
        benchmark_matrix_multiplication(m, n, k);
    }
    return 0;
}

