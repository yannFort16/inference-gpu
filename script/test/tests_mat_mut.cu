// Compile : nvcc -m64 -diag-suppress 2464  -o matrix_mult.exe .\script\test\tests_mat_mut.cu 

#include <stdio.h>
#include <stdlib.h>
#include "../header/mat_mul.h"
#include "../header/utils.h"


int main() {
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
    
        matrix_multiplication(h_A, h_B, h_C1, m, n, k, "default", true);
        
        matrix_multiplication(h_A, h_B, h_C2, m, n, k, "sharedM", true);
    
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
