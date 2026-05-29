#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../operations/mat_mul.cu"

// Function to generate random matrix on host
void generateRandomMatrix(float *matrix, int rows, int cols) {
    //srand(time(NULL));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 10.0;  // Random values between 0 and 10
    }
}

bool compare_matrix(const float *mat1, const float *mat2, int rows, int cols, float epsilon = 1e-3f) {
    int total = rows * cols;
    bool equal = true;
    float max_diff = 0.0f;
    int mismatch_count = 0;

    for (int idx = 0; idx < total; ++idx) {
        float a = mat1[idx];
        float b = mat2[idx];
        float diff = fabsf(a - b);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > epsilon) {
            if (mismatch_count < 5) {
                int row = idx / cols;
                int col = idx % cols;
                printf("Mismatch at [%d,%d]: %f vs %f (diff=%f)\n", row, col, a, b, diff);
            }
            mismatch_count++;
            equal = false;
        }
    }

    if (!equal) {
        printf("Matrix compare failed: %d mismatches, max difference = %f\n", mismatch_count, max_diff);
    } else {
        printf("Matrix compare succeeded: max difference = %f\n", max_diff);
    }

    return equal;
}

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

        generateRandomMatrix(h_A, m, k);
        generateRandomMatrix(h_B, k, n);
    
        matrix_multiplication(h_A, h_B, h_C1, m, n, k, "streamK");
        
        matrix_multiplication(h_A, h_B, h_C2, m, n, k, "default");
    
        bool v = compare_matrix(h_C1, h_C2, m, n, 1e-2f);

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
