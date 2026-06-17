#include <stdio.h>
#include "../header/mat_mul.h"

#define BLOCK_SIZE 8

__device__ float mat_mul_patch(float* input, float* kernel, int k, int n){
    float s = 0.0;
    for(int i = 0; i<k; i++){
        for (int j = 0; j < k; j++){
            s+=input[i*n+j] * kernel[i*k+j];
        }
    }
    return s;
}

__global__ void convolution_default(float* input, int c, int m, int n,
                             float* kernel, int k, int stride,
                             float* output, int nb_patch_w, int nb_patch_h){
    /*
    Simple convolution that is not optimised. Used for referance.
    */
    const int patch_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int patch_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z;
    
    if (patch_row >= nb_patch_h || patch_col >= nb_patch_w ||channel >= c){
        return;    
    }
    

    const int input_offset = channel * m * n;
    const int output_offset = channel * nb_patch_w * nb_patch_h; 
    
    output[output_offset + patch_row*nb_patch_w+patch_col] = 
        mat_mul_patch(&(input[input_offset + patch_row * stride * n + patch_col * stride]),
         &(kernel[channel * k * k]), k, n);

}

__global__ void patch_mat1 (float* input, int c, int m, int n, float *output, int k, int s,
                            int nb_patch_w, int nb_patch_h){
    /* Each thread represents a patch of the matrix and put the content of the patch in the output.
    c -> number of channels
    m -> number of rows in the input
    n -> number of columns in the input
    k -> kernel height and width
    s -> stride
    dim(input) = (c, m, n)
    dim(output) = (c, nb_patch, k)
    output is in row major because it is better for coalescing memory accesses

    */


    const int patch_row = blockIdx.x * blockDim.x + threadIdx.x;
    const int patch_col = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = threadIdx.z;

    int input_offset = channel * m * n;

    const int patch_id = patch_row * nb_patch_w + patch_col;

    if (patch_row < nb_patch_h && patch_col < nb_patch_w && channel < c){
        
        for (int i = 0; i<k; i++){
            for (int j = 0; j<k; j++){
                int pos_in_patch = i*k+j;

                int global_row = patch_row * s + i;
                int global_col = patch_col * s + j;

                output[input_offset + patch_id * k * k + pos_in_patch] = input[input_offset + global_row*n+global_col];
            }
        }
    }
}

__global__ void patch_mat3(float* input, int c, int m, int n, float *output, int k, int s,
                            int nb_patch_w, int nb_patch_h){
    /*
    Each theards represents one element from the patch matrix. They get the corresponding element
    from the input matrix and put it in the output patch matrix.
    */
    const int elem_row = blockIdx.x * blockDim.x + threadIdx.x; // Which Element in the patch
    const int elem_col = blockIdx.y * blockDim.y + threadIdx.y; // Which Patch 
    //const int channel = TODO                       
    
    int patch_row = elem_col/nb_patch_w;
    int patch_col = elem_col%nb_patch_w;

    int kernel_row = elem_row/k;
    int kernel_col = elem_row%k;

    int global_row = patch_row * s + kernel_row;
    int global_col = patch_col * s + kernel_col;

    output[elem_row*nb_patch_w + elem_col] = input[global_row * n + global_col];

    //TODO Matix Multiplication With the kernel
}


__global__ void convolution_shared(float* input, int c, int m, int n, float *output, int k, 
                            float* filter, int s, int nb_patch_w, int nb_patch_h){
    /*
        Convolution where each thread represents an element of the output matrix. They each
        compute the elment they corespond to. First by loading the input in the shared memory then 
        by computing. 

    */
    extern __shared__ float tile[];


    //Coordonates for the elment being calculated in the output matrix
    const int block_patch_row = blockIdx.y * blockDim.y;
    const int block_patch_col = blockIdx.x * blockDim.x;

    const int patch_row = block_patch_row + threadIdx.y;
    const int patch_col = block_patch_col + threadIdx.x;
    const int channel = blockIdx.z;


    const int input_row = block_patch_row * s;
    const int input_col = block_patch_col * s;
    

    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;  
    

    const int channel_input_offset = channel * m * n;
    const int channel_output_offset = channel * nb_patch_w * nb_patch_h;
    const int channel_filter_offset = channel * k * k;

    //Load tile :
    const int tile_size = (BLOCK_SIZE + k-1); 
    for (int i = threadIdx.x; i<tile_size; i+=BLOCK_SIZE){
        int global_row = input_row + i;
        for (int j = threadIdx.y; j<tile_size; j+=BLOCK_SIZE){
            int global_col = input_col + j;
            if (global_col<n && global_row<m){
                tile[i * tile_size + j] = input[channel_input_offset + global_row * n + global_col];
            }else{
                tile[i * tile_size + j] = 0.0;
            }
        }    
    }

    __syncthreads();
    if (patch_col >= nb_patch_w || patch_row >= nb_patch_h || channel >= c) {
        return;
    }

    float sum = 0.0;

    for(int i=0; i<k; i++){
        for(int j=0; j<k; j++){
            sum += tile[(ty+i)* tile_size + tx +j]*filter[channel_filter_offset + i*k + j];
        }
    }
    
    output[channel_output_offset + patch_row * nb_patch_w + patch_col] = sum;
}


float* convolution(float *input, int m, int n, int c, float *filter, int k,
                char* methode, bool pad, int stride, int pad_val){
/* input : input data composed of 2D array over the c chanels
   dim(input) = (c, m, n)
   
   filter : filters to be pas over the input
   dim(kernel) = (c, k, k)

   if pad is true, the padding will be composed of pad_val
*/    
    int nb_patch_w = (n - (k-1) + stride - 1) / stride;
    int nb_patch_h = (m - (k-1) + stride - 1) / stride;
    //int tot_nb_patch = nb_patch_h*nb_patch_w;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid((nb_patch_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (nb_patch_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
                c);

    float *d_input;
    float *d_filter;
    float *d_output;
    
    size_t bytes_input = m*n*c* sizeof(float);
    size_t bytes_filter = k*k*c* sizeof(float);
    size_t bytes_output = nb_patch_w*nb_patch_h*c*sizeof(float);

    cudaMalloc((void**)&d_input, bytes_input);
    cudaMalloc((void**)&d_filter, bytes_filter);
    cudaMalloc((void**)&d_output, bytes_output);


    float* output = (float*)malloc(bytes_output);

    cudaMemcpy(d_input, input, bytes_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, bytes_filter, cudaMemcpyHostToDevice);
    

    if(strcmp(methode, "default") == 0){
        convolution_default<<<grid, block>>>(d_input, c, m, n, d_filter, k, stride, d_output,
                                        nb_patch_w, nb_patch_h);

    }else if (strcmp(methode, "shared") == 0){
        size_t shared_bytes = (BLOCK_SIZE + k-1)*(BLOCK_SIZE + k-1)*sizeof(float);

        convolution_shared<<<grid, block, shared_bytes>>>(d_input, c,  m,  n, d_output, k, 
                            d_filter, stride, nb_patch_w, nb_patch_h);
    }else if (strcmp(methode, "split") == 0){
        //Use kernel to build a patch matrix. Use existing matrix_multiplication
        //TODO    
    }else{
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_filter);
        printf("Methode selected not found. Try default, shared\n");
        exit(1);
    }
    
    cudaMemcpy(output, d_output, bytes_output, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    
    return output;
}