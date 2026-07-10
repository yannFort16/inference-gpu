#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../header/pooling.h"
#include "../header/utils.h"

int main() {
    printf("===== CUDA Pooling Operation =====\n\n");

    // Define matrix dimensions
    int m = 5;   // Input matrix rows
    int n = 5;   // Input matrix columns
    int k = 3;   // Pooling kernel size

    printf("Input matrix dimensions: [%d x %d], Pooling kernel size: %d\n", m, n, k);

    // Allocate host memory
    size_t bytes_input = m * n * sizeof(float);
    float *h_input = (float*)malloc(bytes_input);

    printf("Generating random input matrix...\n");
    generateRandomMatrix(h_input, m, n, 10, true);

    printf("Input matrix:\n");
    printMatrix(h_input, m, n, 6);

    float* output = pooling(h_input, m, n, 1, k, 1, true, 0.0, 'm');

    printf("Pooling operation completed. Results:\n");
    printMatrix(output, m, n, 6);

    free(h_input);
    free(output);
    printf("Done!\n");
    return 0;
}