#include <stdio.h>
#include <stdlib.h>
#include "../header/convolution.h"
#include "../header/utils.h"

int main() {
    printf("===== Test CUDA Convolution =====\n\n");
    
    const int nb_tests = 3; 

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

        // Allocate host memory
        size_t bytes_input = c* m * n * sizeof(float);
        size_t bytes_filter = c * k * k * sizeof(float);
        
        float *h_input = (float*)malloc(bytes_input);
        float *h_filter = (float*)malloc(bytes_filter);
        float *h_output1;
        float *h_output2;
        
        
        for(int i = 0; i<c; i++){
            generateRandomMatrix(&(h_input[i*m*n]), m, n, 50, false);
            generateRandomMatrix(&(h_filter[i*k*k]), k, k, 50, false);
        }
    
        h_output1 = convolution(h_input, m, n, c, h_filter, k, "default", false, 1, 0, true);

        h_output2 = convolution(h_input, m, n, c, h_filter, k, "split", false, 1, 0, true);
        
        bool v =true;
        int out_h = m - k + 1;
        int out_w = n - k + 1;
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
