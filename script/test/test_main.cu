#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "../header/utils.h"
#include "../header/mat_mul.h"

int main(int argc, char **argv) {
    if(argc != 7){
        printf("Usage : %s <input_matrix1> <input_matrix2> <output_matrix> <M> <N> <K>\n", argv[0]);
        return 1;
    }
    printf("Verifying main mat mul validity...\n");

    int m = atoi(argv[4]);
    int n = atoi(argv[5]);
    int k = atoi(argv[6]);
    float alpha = 1.0;
    float beta = 0.0;
    
    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    float *h_C_verif = (float*)malloc(bytes_C);

    int e1, e2, e3 = 0;
    e1 = read_matrix(argv[1], h_A, m, k, 1);
    e2 = read_matrix(argv[2], h_B, k, n, 1);
    e3 = read_matrix(argv[3], h_C_verif, m, n, 1);
    if (e1 != 0 || e2 != 0 || e3 != 0) {
        printf("Error reading or writing matrices.\n");
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_verif);
        return 1;
    }

    matrix_multiplication(h_A, h_B, h_C, m, n, k, "default", false, alpha, beta);

    bool v = compare_matrix(h_C, h_C_verif, m, n);
    if (v) {
        printf("Matrix multiplication is valid.\n");
    } else {
        printf("Matrix multiplication is NOT valid.\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_verif);
    return 0;
}