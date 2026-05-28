#include <stdio.h>
#include <cuda_runtime.h>
//#include <time.h>
#include "../operations/mat_mul.cu"

// Function to generate random matrix on host
void generateRandomMatrix(float *matrix, int rows, int cols) {
    //srand(time(NULL));
    for (int i = 0; i < rows * cols; i++) {
        //matrix[i] = (float)rand() / RAND_MAX * 10.0;  // Random values between 0 and 10
        matrix[i] = rand() % 6; // Random int between 0 and 5
    }
}

// Function to verify results
void printMatrix(const float *matrix, int rows, int cols, int maxElements = 5) {
    /*int min_x = min(rows, maxElements);
    int min_y = min(cols, maxElements);
    for (int i = 0; i< min_x; ++i){
        for(int j = 0; j< min_y; ++j){
            if (i == rows-1 || j == cols-1 || i<=maxElements-2 || j<=maxElements-2 )
            {
                printf("%f ", matrix[i*cols+j]);
            }            else{
                printf("... ");
            }
        }
        printf("\n");
    }*/
    using std::min;

    int visibleRows = min(rows, maxElements);
    int visibleCols = min(cols, maxElements);

    for (int i = 0; i < rows; ++i)
    {
        // Skip middle rows
        if (rows > maxElements + 1 &&
            i >= visibleRows - 1 &&
            i < rows - 1)
        {
            if (i == visibleRows - 1)
                printf("\t...\n");

            continue;
        }

        for (int j = 0; j < cols; ++j)
        {
            // Skip middle columns
            if (cols > maxElements + 1 &&
                j >= visibleCols - 1 &&
                j < cols - 1)
            {
                if (j == visibleCols - 1)
                    printf("... ");

                continue;
            }

            printf("%8.3f ", matrix[i * cols + j]);
        }

        printf("\n");
    }
}

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
    generateRandomMatrix(h_A, m, k);
    generateRandomMatrix(h_B, k, n);
    
    printf("Matrix A sample:\n");
    printMatrix(h_A, m, k, 5);
    printf("Matrix B sample:\n");
    printMatrix(h_B, k, n, 5);
    
    // Allocate device memory and copy data
    printf("\nAllocating device memory and copying data...\n");

    matrix_multiplication(h_A, h_B, h_C, m, n, k, "streamK");
    
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
