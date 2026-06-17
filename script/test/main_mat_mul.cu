//Compile : nvcc -m64 -diag-suppress 2464  -o matrix_mult.exe .\script\test\main_mat_mut.cu 

#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../operation/mat_mul.cu"
#include "../header/utils.h"


int main() {
    printf("===== CUDA Matrix Multiplication =====\n\n");
    
    // Define matrix dimensions
    int m = 4;   // A: m x k
    int n = 3;   // B: k x n, C: m x n
    int k = 2;
    
    printf("Matrix dimensions: A[%d x %d], B[%d x %d], C[%d x %d]\n", m, k, k, n, m, n);
    
    // Allocate host memory
    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);
    
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    
    // Initialize C with zeros
    for (int i = 0; i < m * n; i++) h_C[i] = 0.0;
    
    // Generate random matrices
    printf("Generating random matrices...\n");
    generateRandomMatrix(h_A, m, k, 10, true);
    generateRandomMatrix(h_B, k, n, 10, true);
    
    printf("Matrix A sample:\n");
    printMatrix(h_A, m, k, 5);
    printf("Matrix B sample:\n");
    printMatrix(h_B, k, n, 5);
    
    // Allocate device memory and copy data
    printf("\nAllocating device memory and copying data...\n");

    matrix_multiplication(h_A, h_B, h_C, m, n, k, "sharedM");
    
    // Print results
    printf("\nResult matrix C sample:\n");
    printMatrix(h_C, m, n, 5);
    
    // Cleanup
    printf("\nCleaning up...\n");
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Done!\n");
    return 0;
}
