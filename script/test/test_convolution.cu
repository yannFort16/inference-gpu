#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../header/convolution.h"
#include "../header/utils.h"

int test_convolution(char* methode1, char* methode2) {
    printf("===== Test CUDA Convolution =====\n\n");
    
    const int nb_tests = 3; 

    const bool pad = true;
    const float pad_val = 0.0;
    const int stride = 1;

    int test[nb_tests][4] = {
        {1080, 1920, 3, 3},
        {256, 256, 1, 5},
        {720, 1280, 4, 7},
    }; // Tests with {M, N, C, K}
    
    for(int t = 0; t<nb_tests; t++){
        int m = test[t][0];
        int n = test[t][1];
        int c = test[t][2];
        int k = test[t][3];
        printf("Test with Input dimensions (%d, %d, %d) and kernel (%d, %d, %d)\n",
             c, m, n, c, k, k);

        if(pad){
            printf("padding enabled\n");
        }
        // Allocate host memory
        size_t bytes_input = c* m * n * sizeof(float);
        size_t bytes_filter = c * k * k * sizeof(float);
        
        float *h_input = (float*)malloc(bytes_input);
        float *h_filter = (float*)malloc(bytes_filter);
        float *h_output1 = (float*)malloc(bytes_input);
        float *h_output2 = (float*)malloc(bytes_input);;
        
        
        for(int i = 0; i<c; i++){
            generateRandomMatrix(&(h_input[i*m*n]), m, n, 50, false);
            generateRandomMatrix(&(h_filter[i*k*k]), k, k, 50, false);
        }
    

        convolution(h_input, h_output1, m, n, c, h_filter, k, methode1, false, stride, pad_val, pad);

        convolution(h_input, h_output2, m, n, c, h_filter, k, methode2, false, stride, pad_val, pad);
        
        bool v = true;
        int out_h = (pad ? m - k + 1 : m)/stride;
        int out_w = (pad ? n - k + 1 : n)/stride;
        int channel_size = out_h * out_w;
        for (int i = 0; i < c; i++) {
            v = v && compare_matrix(
                h_output1 + i * channel_size,
                h_output2 + i * channel_size,
                out_h, out_w);
        }

        if (v) {
            printf("Test %d passed\n", t);
        } else {
            printf("Test %d FAILED\n", t);
        }
        printf("\n----------------------------------\n");
        free(h_input);
        free(h_filter);
        free(h_output1);
        free(h_output2);
    }
    printf("Done!\n");
    return 0;
}

int benchmark_convolution(char* methode1, char* methode2, int m, int n, int k) {
    printf("===== Benchmark CUDA Convolution =====\n\n");
    
    const int loop_count = 32; 

    printf("Benchmark with Matrix dimensions [%d x %d] and filter [%d x %d]\n", m, n, k, k);


    // Allocate host memory
    size_t bytes_input = m * n * sizeof(float);
    size_t bytes_filter = k * k * sizeof(float);
    size_t bytes_output = (m-k+1) * (n-k+1) * sizeof(float);
        
    float *input = (float*)malloc(bytes_input);
    float *filter = (float*)malloc(bytes_filter);
    float *res1 = (float*)malloc(bytes_output);
    float *res2 = (float*)malloc(bytes_output);

    cudaEvent_t begining1, finish1, begining2, finish2;
    cudaEventCreate(&begining1);
    cudaEventCreate(&finish1);
    cudaEventCreate(&begining2);
    cudaEventCreate(&finish2);
    
    bool last_iter = false;
    
    for(int t = 0; t<loop_count; t++){    
        generateRandomMatrix(input, m, n, 50, false);
        generateRandomMatrix(filter, k, k, 50, false);
        
        last_iter = (t == loop_count - 1);
        
        if (last_iter)
            cudaEventRecord(begining1);

        convolution(input, res1, m, n, 1, filter, k, methode1, false, 1, 0, false);
        if (last_iter)
            cudaEventRecord(finish1);

        //Wait before starting the second convolution to avoid overlapping events
        cudaDeviceSynchronize();

        if (last_iter)
            cudaEventRecord(begining2);
        convolution(input, res2, m, n, 1, filter, k, methode2, false, 1, 0, false);
        if (last_iter)
            cudaEventRecord(finish2);

        if (last_iter) {
            compare_matrix(res1, res2, m-k+1, n-k+1); 
        }
        //printf("%d/%d iterations completed\n", t+1, loop_count);
    }

    float time1, time2;
    cudaEventSynchronize(finish1);
    cudaEventSynchronize(finish2);
    cudaEventElapsedTime(&time1, begining1, finish1);
    cudaEventElapsedTime(&time2, begining2, finish2);
    printf("Time for %s: %f ms\n", methode1, time1);
    printf("Time for %s: %f ms\n", methode2, time2);


    free(input);
    free(filter);
    free(res1);
    free(res2);
    printf("Done!\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <methode1> <methode2> <t || b>\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[3], "t") == 0) {
        return test_convolution(argv[1], argv[2]);
    } else if (strcmp(argv[3], "b") == 0) {
        
        int m = 12288;
        int n = 4016;
        int k = 17;
        return benchmark_convolution(argv[1], argv[2], m, n, k);
    } else {
        printf("Unknown command: %s\nUse <t> or <b>", argv[3]);
        return 1;
    }
}