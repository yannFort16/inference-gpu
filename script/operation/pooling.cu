#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "../header/pooling.h"

#define BLOCK_SIZE 16

__device__ float max_pool(float *input, int m, int n, int start_row, int start_col,
                            int k, int stride, bool padding, float pad_val){

    float max_val = -FLT_MAX;

    for(int i = 0; i < k; i++){
        int row = start_row + i;
        for(int j = 0; j < k; j++){
            int col = start_col + j;

            if(row >= 0 && row < m && col >= 0 && col < n){
                float val = input[row * n + col];
                if(val > max_val){
                    max_val = val;
                }
            } else if(padding){
                if(pad_val > max_val){
                    max_val = pad_val;
                }
            }
        }
    }
    
    return max_val;
}

__device__ float avg_pool(float *input, int m, int n, int start_row, int start_col,
                            int k, int stride, bool padding, float pad_val){

    float sum = 0.0f;
    int count = 0;

    for(int i = 0; i < k; i++){
        for(int j = 0; j < k; j++){
            int row = start_row + i;
            int col = start_col + j;

            if(row >= 0 && row < m && col >= 0 && col < n){
                sum += input[row * n + col];
                count++;
            } else if(padding){
                sum += pad_val;
                count++;
            }
        }
    }

    return count > 0 ? sum / count : 0.0f;
}

__global__ void pooling(float *input, float *output, int m, int n, int channels,
                            int k, int nb_patch_w, int nb_patch_h, int stride, 
                            bool padding, float pad_val, char pooling_type) {

    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row >= nb_patch_h || out_col >= nb_patch_w) 
        return;

    const int channel = blockIdx.z;

    const int input_offset = channel * m * n;
    const int output_offset = channel * nb_patch_w * nb_patch_h; 
    
    int start_row = out_row * stride - (padding ? (k - 1) / 2 : 0);
    int start_col = out_col * stride - (padding ? (k - 1) / 2 : 0);

    float result = 0.0f;
    if (pooling_type == 'm') {
        result = max_pool(input + input_offset, m, n, start_row, start_col, k, stride, padding, pad_val);
    } else if (pooling_type == 'a') {
        result = avg_pool(input + input_offset, m, n, start_row, start_col, k, stride, padding, pad_val);
    }
    output[output_offset + out_row * nb_patch_w + out_col] = result;

}


float * pooling(float *input, int m, int n, int channels, int k, int stride, 
                bool padding, float pad_val, char pooling_type) {
    /* 
    pooling_type is either : m for max || a for average

    input : input data composed of 2D array over the c chanels
    dim(input) = (c, m, n)
   
    k : size of the kernel to be pas over the input
    stride : stride of the kernel
    padding : if true, the padding will be composed of pad_val. Works only for k odd.
    */

    if (pooling_type != 'm' && pooling_type != 'a') {
        fprintf(stderr, "Error: pooling_type must be either 'm' for max or 'a' for average.\n");
        return NULL;
    }

    int dim_m = padding ? m + 2 * ((k - 1) / 2) : m;
    int dim_n = padding ? n + 2 * ((k - 1) / 2) : n;

    int nb_patch_w = (dim_n - (k-1) + stride - 1) / stride;
    int nb_patch_h = (dim_m - (k-1) + stride - 1) / stride;
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((nb_patch_w + blockDim.x - 1) / blockDim.x,
                 (nb_patch_h + blockDim.y - 1) / blockDim.y,
                 channels);

    size_t bytes_input = m*n*channels* sizeof(float);
    size_t bytes_output = nb_patch_w*nb_patch_h*channels*sizeof(float);

    float* output = (float*)malloc(bytes_output);

    float *d_input;
    float *d_output;

    cudaMalloc((void**)&d_input, bytes_input);
    cudaMalloc((void**)&d_output, bytes_output);

    cudaMemcpy(d_input, input, bytes_input, cudaMemcpyHostToDevice);
    

    pooling<<<gridDim, blockDim>>>(d_input, d_output, m, n, channels,
                                    k, nb_patch_w, nb_patch_h,
                                    stride, padding, pad_val,
                                    pooling_type);
    
    cudaMemcpy(output, d_output, bytes_output, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}