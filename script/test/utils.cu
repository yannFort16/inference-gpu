#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <stdbool.h>


#define EPSILON 1e-3f

// Function to generate random matrix on host
void generateRandomMatrix(float *matrix, int rows, int cols, float max, bool whole) {
    srand(time(NULL));
    for (int i = 0; i < rows * cols; i++) {
        if (whole) {
            matrix[i] = (float)(rand() % (int)max); // Random int between 0 and max
        } else {
            matrix[i] = (float)rand() / RAND_MAX * max; // Random float between 0 and max
        }
    }
}

// Function to verify results
void printMatrix(const float *matrix, int rows, int cols, int maxElements) {
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
    //using std::min;

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

bool compare_matrix(const float *mat1, const float *mat2, int rows, int cols) {
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
        if (diff > EPSILON) {
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
    }/* else {
        printf("Matrix compare succeeded: max difference = %f\n", max_diff);
    }*/

    return equal;
}

int read_matrix(char * filename, float* dest_matrix, int x, int y, int c) {
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "read_matrix: could not open file %s\n", filename);
        return -1;
    }

    size_t num_elements = (size_t)x * y * c;
    size_t read_count = fread(dest_matrix, sizeof(float), num_elements, fp);
    fclose(fp);

    if (read_count != num_elements) {
        fprintf(stderr, "read_matrix: expected %zu elements, got %zu\n",
                num_elements, read_count);
        return -1;
    }
    return 0;
}

int write_matrix(char * filename, float* src_matrix, int x, int y, int c) {
    FILE* fp = fopen(filename, "wb");
    if (fp == NULL) {
        fprintf(stderr, "write_matrix: could not open file %s\n", filename);
        return -1;
    }

    size_t num_elements = (size_t)x * y * c;
    size_t write_count = fwrite(src_matrix, sizeof(float), num_elements, fp);
    fclose(fp);

    if (write_count != num_elements) {
        fprintf(stderr, "write_matrix: expected to write %zu elements, wrote %zu\n",
                num_elements, write_count);
        return -1;
    }
    return 0;
}