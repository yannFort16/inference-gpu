#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "header/mat_mul.h"
#include "header/convolution.h"
#include "header/pooling.h"
#include "header/utils.h"


int main_mat_mul(int argc, char **argv){
    int m = atoi(argv[5]);
    int n = atoi(argv[6]);
    int k = atoi(argv[7]);
    float alpha = atof(argv[8]);
    float beta = atof(argv[9]);

    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    int e1, e2, e3 =0;
    if (strcmp(argv[10], "-g") == 0) {
        // Generate random matrices and perform multiplication
        generateRandomMatrix(h_A, m, k, 50, false);
        generateRandomMatrix(h_B, k, n, 50, false);

        e1 = write_matrix(argv[2], h_A, m, k, 1);
        e2 = write_matrix(argv[3], h_B, k, n, 1);
    } else if (argc == 11) {
        // Read matrices from files and perform multiplication
        e1 = read_matrix(argv[2], h_A, m, k, 1);
        e2 = read_matrix(argv[3], h_B, k, n, 1);
        e3 = read_matrix(argv[4], h_C, m, n, 1); // Read the output matrix if it exists
    }else {
        printf("Invalid number of arguments for matrix multiplication.\n");
        return 1;
    }
    if (e1 != 0 || e2 != 0 || e3 != 0) {
        printf("Error reading or writing matrices.\n");
        free(h_A);
        free(h_B);
        free(h_C);
        return 1;
    }
    matrix_multiplication(h_A, h_B, h_C, m, n, k, "sharedM", false, alpha, beta);

    e1 = write_matrix(argv[4], h_C, m, n, 1);
    if (e1 != 0) {
        printf("Error writing output matrix.\n");
        free(h_C);
        free(h_A);
        free(h_B);
        return 1;
    }

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}

int main_convolution_pooling(int argc, char **argv){
    int m = atoi(argv[5]);
    int n = atoi(argv[6]);
    int k = atoi(argv[7]);
    int c = atoi(argv[8]);
    int stride = atoi(argv[9]);
    bool padding = strcmp(argv[10], "true") == 0;
    float padding_value = atof(argv[11]);

    size_t bytes_input = m * n * c * sizeof(float);
    size_t bytes_filter = k * k * c * sizeof(float);

    float *h_input = (float*)malloc(bytes_input);
    float *h_filter = (float*)malloc(bytes_filter);
    float *h_output = NULL;

    int e1, e2 = 0;
    if(strcmp(argv[12], "-g") == 0) {
        generateRandomMatrix(h_input, m*c, n, 50, false);
        generateRandomMatrix(h_filter, k*c, k, 50, false);

        e1 = write_matrix(argv[2], h_input, m, n, c);
        e2 = write_matrix(argv[3], h_filter, k, k, c);
    } else if (argc == 13) {
        e1 = read_matrix(argv[2], h_input, m, n, c);
        e2 = read_matrix(argv[3], h_filter, k, k, c);
    } else {
        printf("Invalid number of arguments for convolution/pooling.\n");
        return 1;
    }
    if (e1 != 0 || e2 != 0) {
        printf("Error reading or writing matrices.\n");
        free(h_input);
        free(h_filter);
        return 1;
    }

    int output_height, output_width;
    get_output_dimensions(m, n, k, stride, padding, &output_height, &output_width);
    size_t bytes_output = output_height * output_width * c * sizeof(float);
    h_output = (float*)malloc(bytes_output);

    if (strcmp(argv[1], "convolution") == 0) {
        convolution(h_input, h_output, m, n, c, h_filter, k, "default", padding, stride, padding_value, false);
    } else if (strcmp(argv[1], "pooling") == 0) {
        pooling(h_input, h_output, m, n, c, k, stride, padding, padding_value, 'm');
    }

    e1 = write_matrix(argv[4], h_output, output_height, output_width, c);
    if (e1 != 0) {
        printf("Error writing output matrix.\n");
        free(h_output);
        free(h_input);
        free(h_filter);
        return 1;
    }
    free(h_output);
    free(h_input);
    free(h_filter);
    return 0;
}

void print_help(char *argv0){
    printf("Usage: %s <operation> [options]\n\n", argv0);
    printf("Operations:\n");
    printf("  - multiplication\n");
    printf("  - convolution\n");
    printf("  - pooling\n\n");
    printf("Matrix Multiplication:\n");
    printf("  %s multiplication <input_matrix1> <input_matrix2> <output_matrix> <M> <N> <K> <alpha> <beta> -g\n\n", argv0);
    printf("Convolution:\n");
    printf("  %s convolution <input_matrix> <filter> <output_matrix> <M> <N> <K> <channels> <stride> <padding> <padding_value> -g\n\n", argv0);
    printf("Pooling:\n");
    printf("  %s pooling <input_matrix> <filter> <output_matrix> <M> <N> <K> <channels> <stride> <padding> <padding_value> -g\n\n", argv0);
    printf("Options:\n");
    printf("  -g: Generate random input matrices\n");
}

int main(int argc, char **argv){
    if (argc <= 9){
        print_help(argv[0]);
        return 1;
    }
    if(strcmp(argv[1], "multiplication") == 0 && 10<=argc && argc<=11){
        return main_mat_mul(argc, argv);
    }if((strcmp(argv[1], "convolution") == 0 || strcmp(argv[1], "pooling") == 0 ) && 12<=argc && argc<=13){
        return main_convolution_pooling(argc, argv);
    } else {
        printf("Invalid operation or number of arguments.\n");
        print_help(argv[0]);
    }

    return 0;
}

