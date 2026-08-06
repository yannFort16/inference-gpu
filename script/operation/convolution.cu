#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "../header/mat_mul.h"

#define BLOCK_SIZE 16
#define MAX_FILTER_SIZE 16384

__constant__ float const_filter[MAX_FILTER_SIZE]; 

__device__ float mat_mul_patch(float* input, float* filter, int k, int n, int m,
                              int start_row, int start_col, bool use_const_filter,
                              bool pad, float pad_val) {
    float* const_filter_ptr = use_const_filter ? const_filter : filter;
    float s = 0.0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            int row = start_row + i;
            int col = start_col + j;
            float input_val;
            if (pad && (row < 0 || row >= m || col < 0 || col >= n)) {
                input_val = pad_val;
            } else {
                input_val = input[row * n + col];
            }
            s += input_val * const_filter_ptr[i * k + j];
        }
    }
    return s;
}

__global__ void convolution_default(float* input, int c, int m, int n,
                             float* filter, int k, int stride,
                             float* output, int nb_patch_w, int nb_patch_h, bool pad, float pad_val){
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
    
    bool use_const_filter = (k*k*c < MAX_FILTER_SIZE);
    int start_row = patch_row * stride - (pad ? (k - 1) / 2 : 0);
    int start_col = patch_col * stride - (pad ? (k - 1) / 2 : 0);

    output[output_offset + patch_row*nb_patch_w+patch_col] = 
        mat_mul_patch(input + input_offset, filter, k, n, m,
          start_row, start_col, use_const_filter, pad, pad_val);

}

__global__ void patch_mat1 (float* input, int c, int m, int n, float *output, int k, int s,
                            int nb_patch_w, int nb_patch_h){
    /* Each thread represents a patch of the matrix and puts the content of the patch in the output.
    c -> number of channels
    m -> number of rows in the input
    n -> number of columns in the input
    k -> kernel height and width
    s -> stride
    dim(input) = (c, m, n)
    dim(output) = (c, nb_patch, k*k)
    output is in row major because it is better for coalescing memory accesses
    */

    const int patch_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int patch_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = threadIdx.z;

    int input_offset = channel * m * n;
    const int patch_id = patch_row * nb_patch_w + patch_col;

    if (patch_row < nb_patch_h && patch_col < nb_patch_w && channel < c){
        for (int i = 0; i < k; i++){
            for (int j = 0; j < k; j++){
                int pos_in_patch = i * k + j;
                int global_row = patch_row * s + i;
                int global_col = patch_col * s + j;
                output[input_offset + patch_id * k * k + pos_in_patch] = input[input_offset + global_row * n + global_col];
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
    const int patch_id = blockIdx.x * blockDim.x + threadIdx.x; // Which Pathc
    const int kernel_elem = blockIdx.y * blockDim.y + threadIdx.y; //Which Element in the patch
    const int channel = blockIdx.z;                       
    
    int patch_row = patch_id/nb_patch_w;
    int patch_col = patch_id%nb_patch_w;
    int tot_nb_patch = nb_patch_w*nb_patch_h;

    if (patch_id >= tot_nb_patch || kernel_elem >= k*k)
        return;

    int kernel_row = kernel_elem/k;
    int kernel_col = kernel_elem%k;

    int global_row = patch_row * s + kernel_row;
    int global_col = patch_col * s + kernel_col;

    output[channel*(k*k)*tot_nb_patch + kernel_elem*tot_nb_patch + patch_id] = 
            input[channel*m*n + global_row * n + global_col];

    //TODO Matix Multiplication With the kernel
}

__global__ void convolution_tiled(
    const float* __restrict__ input,
    const float* __restrict__ filter,
    float* __restrict__ output,
    int c, int m, int n,
    int k, int stride,
    int out_w, int out_h,
    bool pad, float pad_val)
    //USELESS FOR NOW. USE convolution_shared instead
{
    extern __shared__ float tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int channel = blockIdx.z;

    const int out_row = blockIdx.y * BLOCK_SIZE + ty;
    const int out_col = blockIdx.x * BLOCK_SIZE + tx;

    const int pad_radius = pad ? (k - 1) / 2 : 0;

    const int tile_size = BLOCK_SIZE * stride + k - 1;

    const int in_row0 = blockIdx.y * BLOCK_SIZE * stride - pad_radius;
    const int in_col0 = blockIdx.x * BLOCK_SIZE * stride - pad_radius;

    const int input_offset  = channel * m * n;
    const int output_offset = channel * out_h * out_w;
    const int filter_offset = channel * k * k;

    const float* f = (c * k * k < MAX_FILTER_SIZE)
                     ? const_filter
                     : filter;

    int linear_tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;
    int tile_elems = tile_size * tile_size;

    for (int idx = linear_tid; idx < tile_elems; idx += num_threads) {
        int tr = idx / tile_size;
        int tc = idx % tile_size;

        int gr = in_row0 + tr;
        int gc = in_col0 + tc;

        float v = pad_val;
        if (gr >= 0 && gr < m && gc >= 0 && gc < n) {
            v = input[input_offset + gr * n + gc];
        }
        tile[idx] = v;
    }

    __syncthreads();

    if (out_row < out_h && out_col < out_w) {
        float sum = 0.0f;

        int base_r = ty * stride;
        int base_c = tx * stride;

        #pragma unroll
        for (int i = 0; i < 7; i++) {
            if (i >= k) break;
            #pragma unroll
            for (int j = 0; j < 7; j++) {
                if (j >= k) break;
                sum += tile[(base_r + i) * tile_size + (base_c + j)] *
                       f[filter_offset + i * k + j];
            }
        }

        output[output_offset + out_row * out_w + out_col] = sum;
    }
}


__global__ void convolution_shared_wt_appron(float* input, int c, int m, int n, float *output, const float* filter,
                                 int k){ //, int s, int nb_patch_w, int nb_patch_h, bool padding, float pad_val
    /*
        Convolution where each thread represents an element of the output matrix. They each
        compute the elment they corespond to. First by loading the input in the shared memory then 
        by computing. 

        No padding implemented yet

        USLESS FOR NOW. USE convolution_shared instead
    */
    extern __shared__ float s_tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * BLOCK_SIZE + tx;
    int gy = blockIdx.y * BLOCK_SIZE + ty;

    int TILE_SIZE = BLOCK_SIZE + k - 1;

    // Load Phase: Coalesced read
    for (int load_y = ty; load_y < TILE_SIZE; load_y += blockDim.y)
    {
        for (int load_x = tx; load_x < TILE_SIZE; load_x += blockDim.x)
        {

            // Calculate global coordinates for the pixel to load
            int input_x = blockIdx.x * BLOCK_SIZE - (k - 1) / 2 + load_x;
            int input_y = blockIdx.y * BLOCK_SIZE - (k - 1) / 2 + load_y;

            if (input_x >= 0 && input_x < m && input_y >= 0 && input_y < n)
            {
                s_tile[load_y * TILE_SIZE + load_x] = input[input_y * n + input_x];
            }
            else
            {
                s_tile[load_y * TILE_SIZE + load_x] = 0;
            }
        }
    }

    // Sync threads
    __syncthreads();

    // Compute Phase: Using shared memory
    if (gx < n && gy < m)
    {
        float sum = 0.0f;
        for (int kRow = 0; kRow < k; ++kRow)
        {
            for (int kCol = 0; kCol < k; ++kCol)
            {
                sum += s_tile[(ty + kRow) * TILE_SIZE + (tx + kCol)] * filter[kRow * k + kCol];
            }
        }

        output[gy * n + gx] = sum;
    }

    return;
}

__global__ void convolution_shared(float* input, int c, int m, int n, float *output, float* filter,
                                 int k, int s, int nb_patch_w, int nb_patch_h, bool padding, float pad_val){
    /*
        Convolution where each thread represents an element of the output matrix. They each
        compute the elment they corespond to. First by loading the input in the shared memory then 
        by computing. 

        Added padding option to the convolution. If padding is true, the input will be padded with pad_val.
        Works only for k odd.
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
    
    float* const_filter_ptr = c * k * k < MAX_FILTER_SIZE ? const_filter : filter; // 
    
    //Load tile :
    const int tile_size = (BLOCK_SIZE * s + k-1); 
    for (int row = threadIdx.y; row<tile_size; row+=BLOCK_SIZE){
        int global_row = padding ? input_row + row - (k - 1)/2 : input_row + row;
        
        for (int col = threadIdx.x; col<tile_size; col+=BLOCK_SIZE){
            int global_col = padding ? input_col + col - (k - 1)/2 : input_col + col;
        
            if (global_col<n && global_row<m && global_col>=0 && global_row>=0){
                tile[row * tile_size + col] = input[channel_input_offset + global_row * n + global_col];
            }else{
                tile[row * tile_size + col] = pad_val;
            }
        }    
    }

    //Coalesced memory load :
    //TODO

    __syncthreads();
    if (patch_col >= nb_patch_w || patch_row >= nb_patch_h || channel >= c) {
        return;
    }

    float sum = 0.0;

    for(int i=0; i<k; i++){
        for(int j=0; j<k; j++){
            sum += tile[(ty*s+i)* tile_size + tx*s +j]*const_filter_ptr[channel_filter_offset + i*k + j];
        }
    }
    
    output[channel_output_offset + patch_row * nb_patch_w + patch_col] = sum;
}


void convolution(float *input, float *output, int m, int n, int c, float *filter, int k,
                char* methode, bool pad, int stride, float pad_val, bool benchmark){
/* input : input data composed of 2D array over the c chanels
   dim(input) = (c, m, n)
   
   filter : filters to be pas over the input
   dim(kernel) = (c, k, k)

   if pad is true, the padding will be composed of pad_val
*/  
    int dim_m = pad ? m + 2 * ((k - 1) / 2) : m;
    int dim_n = pad ? n + 2 * ((k - 1) / 2) : n;

    int nb_patch_w = (dim_n - (k-1) + stride - 1) / stride;
    int nb_patch_h = (dim_m - (k-1) + stride - 1) / stride;
    //int tot_nb_patch = nb_patch_h*nb_patch_w;

    bool use_const_filter = (k*k*c < MAX_FILTER_SIZE);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid((nb_patch_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (nb_patch_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
                c);
    int tile_size = (BLOCK_SIZE * stride + k-1);
    size_t shared_bytes = (tile_size * tile_size ) * sizeof(float); //+(k-1)


    float *d_input;
    float *d_filter;
    float *d_output;
    
    size_t bytes_input = m*n*c* sizeof(float);
    size_t bytes_filter = k*k*c* sizeof(float);
    size_t bytes_output = nb_patch_w*nb_patch_h*c*sizeof(float);

    cudaMalloc((void**)&d_input, bytes_input);
    cudaMalloc((void**)&d_filter, bytes_filter);
    cudaMalloc((void**)&d_output, bytes_output);

    // Events for CUDA timing
    cudaEvent_t start, afterH2D, afterKernel, afterD2H;
    if(benchmark){
        cudaEventCreate(&start);
        cudaEventCreate(&afterH2D);
        cudaEventCreate(&afterKernel);
        cudaEventCreate(&afterD2H);
    }

    if (benchmark) cudaEventRecord(start);
    cudaMemcpy(d_input, input, bytes_input, cudaMemcpyHostToDevice);
    if (use_const_filter) {//
        cudaMemcpyToSymbol(const_filter, filter, bytes_filter);
    } else {
        cudaMemcpy(d_filter, filter, bytes_filter, cudaMemcpyHostToDevice);
    }
    

    if(strcmp(methode, "default") == 0){
        if (benchmark) cudaEventRecord(afterH2D);
        convolution_default<<<grid, block>>>(d_input, c, m, n, d_filter, k, stride, d_output,
                                        nb_patch_w, nb_patch_h, pad, pad_val);
        /*cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("CUDA error after convolution_default kernel: %s\n", cudaGetErrorString(e));
            exit(1);
        } 
        cudaDeviceSynchronize();*/

        if (benchmark) cudaEventRecord(afterKernel);

        cudaMemcpy(output, d_output, bytes_output, cudaMemcpyDeviceToHost);
        if (benchmark) cudaEventRecord(afterD2H);
    }else if (strcmp(methode, "shared") == 0){
        size_t shared_bytes = (BLOCK_SIZE + k-1)*(BLOCK_SIZE + k-1)*sizeof(float);

        if (benchmark) cudaEventRecord(afterH2D);
        convolution_shared<<<grid, block, shared_bytes>>>(d_input, c,  m,  n, d_output, d_filter, k, 
                             stride, nb_patch_w, nb_patch_h, pad, pad_val);
        /*cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("CUDA error after convolution_shared kernel: %s\n", cudaGetErrorString(e));
            exit(1);
        }
        cudaDeviceSynchronize();*/
        
        if (benchmark) cudaEventRecord(afterKernel);

        cudaMemcpy(output, d_output, bytes_output, cudaMemcpyDeviceToHost);
        if (benchmark) cudaEventRecord(afterD2H);
        
    }else if (strcmp(methode, "tiled") == 0){
        if (benchmark) cudaEventRecord(afterH2D);
        convolution_shared_wt_appron<<<grid, block, shared_bytes>>>(d_input, c,  m,  n, d_output, d_filter, k);
        if (benchmark) cudaEventRecord(afterKernel);

        cudaMemcpy(output, d_output, bytes_output, cudaMemcpyDeviceToHost);
        if (benchmark) cudaEventRecord(afterD2H);
    }else if (strcmp(methode, "split") == 0 || strcmp(methode, "ref") == 0){
        
        // Build a patch matrix and compute per-channel convolution as a matrix multiplication.
        int nb_patches = nb_patch_h * nb_patch_w;
        size_t bytes_patch_mat = c * nb_patches * k * k * sizeof(float);
        float* patch_mat = (float*)malloc(bytes_patch_mat);
        float* d_patch_mat;
        cudaMalloc((void**)&d_patch_mat, bytes_patch_mat);

        dim3 gridPatch((nb_patches + BLOCK_SIZE - 1) / BLOCK_SIZE,
               ((k*k) + BLOCK_SIZE - 1) / BLOCK_SIZE,
               c);

        if (benchmark) cudaEventRecord(afterH2D);
        patch_mat3<<<gridPatch, block>>>(d_input, c, m, n, d_patch_mat, k, stride, nb_patch_w, nb_patch_h);
        
        if (benchmark) cudaEventRecord(afterKernel);

        cudaMemcpy(patch_mat, d_patch_mat, bytes_patch_mat, cudaMemcpyDeviceToHost);
        cudaFree(d_patch_mat);

        bool is_ref = (strcmp(methode, "ref") == 0);
        cublasStatus_t stat;
        cublasHandle_t handle;
        if(is_ref){
            stat = cublasCreate(&handle);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("CUBLAS initialization failed (code %d)\n", stat);
                exit(1);
            }
        }
        //Matrix multiplication
        for (int channel = 0; channel < c; channel++) {
            float* channel_filter = filter + channel * k * k;
            float* channel_patch_mat = patch_mat + channel * nb_patches * k * k;
            float* channel_output = output + channel * nb_patches;
            if (is_ref){
            matrix_multiplication(channel_filter, channel_patch_mat, channel_output,
                                  1, nb_patches, k * k, "sharedM", false);
        
            } else {          
                float alpha = 1;
                float beta = 0;   
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_filter, n, d_patch_mat, k, &beta, d_output, n);
                cudaDeviceSynchronize();
            }
        }
        free(patch_mat);
        if (benchmark) cudaEventRecord(afterD2H);
        if (is_ref) {
            cublasDestroy(handle);
        }
    }else{
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_filter);
        printf("Methode selected not found. Try default or shared\n");
        exit(1);
    }
    
    
    if(benchmark){
        cudaEventSynchronize(afterD2H);
        float h2d_ms, kernel_ms, d2h_ms, total_ms;
        cudaEventElapsedTime(&h2d_ms, start, afterH2D);
        cudaEventElapsedTime(&kernel_ms, afterH2D, afterKernel);
        cudaEventElapsedTime(&d2h_ms, afterKernel, afterD2H);
        cudaEventElapsedTime(&total_ms, start, afterD2H);

        print_performance(h2d_ms, kernel_ms, d2h_ms, total_ms, methode);
        
        cudaEventDestroy(start);
        cudaEventDestroy(afterH2D);
        cudaEventDestroy(afterKernel);
        cudaEventDestroy(afterD2H);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    
    
    
    return;
}

void get_output_dimensions(int m, int n, int k, int stride, bool pad, int* out_m, int* out_n) {
    /* Calculate the output dimensions based on the input dimensions, filter size, stride, and padding */
    if (pad) {
        *out_m = m;
        *out_n = n;
    } else {
        *out_m = (m - (k - 1) + stride - 1) / stride;
        *out_n = (n - (k - 1) + stride - 1) / stride;
    }
}

