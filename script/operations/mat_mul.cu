#include <stdio.h>
#include <string.h>

#define BLK_M 64 //Tile dim in M
#define BLK_N 64 //Tile dim in N
#define BLK_K 32 //Tile dim in K

#define BLOCK_SIZE 32
#define THREAD_TILE 4

__device__ inline int ceil_div(int a, int b){
    return (a + b - 1) / b;
}

__global__ void point(float *A, float * B, float * C,
                        float alpha, float beta, 
                        int m, int n, int k ){

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.y;
    if (row < m && col < n){
        float tmp = 0.0;

        for (int i = 0; i<k; i++){
            tmp += A[row * k +i] * B[i * n + col];
        }

        C[row * n + col] = alpha * C[row * n + col] + beta * tmp;
    } 

}


__global__ void point_shared(float *A, float * B, float * C,
                        int m, int n, int k ){
    
    __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int g_row = blockIdx.x * BLOCK_SIZE + ty * THREAD_TILE;
    const int g_col = blockIdx.y * BLOCK_SIZE + tx * THREAD_TILE;

    /*float sum = 0.0;*/
    float accum[THREAD_TILE][THREAD_TILE] = {0.0};

    /*==============LOOP OVER THE TILES==============*/
    const int nb_tiles = ceil_div(k,BLOCK_SIZE);
    for(int t = 0; t<nb_tiles; t++){

        const int t_col_A = t * BLOCK_SIZE + tx * THREAD_TILE;
        const int t_row_B = t * BLOCK_SIZE + ty * THREAD_TILE;

        /*if (g_row < m && t_col_A < k){
            tile_A[ty][tx] = A[g_row * k + t_col_A];
        }else{
            tile_A[ty][tx] = 0;
        }

        if (g_col < n && t_row_B < k){
            tile_B[ty][tx] = B[t_row_B * n + g_col];
        }else{
            tile_B[ty][tx] = 0;
        }*/
        for (int v = 0; v < THREAD_TILE; v++) {
            int row = g_row + v;
            for (int u = 0; u < THREAD_TILE; u++) {
                int colA = t_col_A + u;
                int sharedRow = ty * THREAD_TILE + v;
                int sharedCol = tx * THREAD_TILE + u;
                tile_A[sharedRow][sharedCol] = (row < m && colA < k)
                    ? A[row * k + colA]
                    : 0.0f;
            }
        }

        for (int v = 0; v < THREAD_TILE; v++) {
            int rowB = t_row_B + v;
            for (int u = 0; u < THREAD_TILE; u++) {
                int col = g_col + u;
                int sharedRow = ty * THREAD_TILE + v;
                int sharedCol = tx * THREAD_TILE + u;
                tile_B[sharedRow][sharedCol] = (rowB < k && col < n)
                    ? B[rowB * n + col]
                    : 0.0f;
            }
        }

        __syncthreads();

        /*for(int i = 0; i < BLOCK_SIZE; i++){
            sum+= tile_A[ty][i] * tile_B[i][tx];
        }*/
        for(int i = 0; i < BLOCK_SIZE; i++){
            float regA[THREAD_TILE];
            float regB[THREAD_TILE];

            for(int v =0; v<THREAD_TILE; v++){
                //Load A
                regA[v] = tile_A[ty * THREAD_TILE + v][i];
                //Load B
                regB[v] = tile_B[i][tx * THREAD_TILE + v];
            }

            for (int v = 0; v < THREAD_TILE; v++){
                for (int w = 0; w < THREAD_TILE; w++){
                    accum[v][w] += regA[v] * regB[w];
                }
            }
        }
        

        __syncthreads();
    }

    for (int v = 0; v < THREAD_TILE; v++){
        for (int w = 0; w < THREAD_TILE; w++){
            int row = g_row + v;
            int col = g_col + w;
            if (row < m && col < n){
                C[row * n + col] = accum[v][w];
            }
        }
    }
}

__global__ void streamK_point(float *A, float * B, float * C,
                               int* flags, float * partials, 
                                int m, int n, int k){
    __shared__ float accum[BLK_M][BLK_N];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int iters_per_tile = ceil_div(k, BLK_K);
    const int tile_m = ceil_div(m, BLK_M);
    const int tile_n = ceil_div(n, BLK_N);

    const int total_iters = tile_m * tile_n * iters_per_tile;
    
    int num_ctas = gridDim.x * gridDim.y;
    int cta_id = blockIdx.y * gridDim.x + blockIdx.x;
    int iter_per_cta = ceil_div(total_iters, num_ctas);
    
    //Instantiate CTAs
    int iter = cta_id * iter_per_cta;
    int iter_end = min(iter + iter_per_cta, total_iters);

    //Iteration processing outer loop
    while (iter < iter_end){
        int tile_idx = iter / iters_per_tile;
        int tile_iter = tile_idx * iters_per_tile;
        int tile_iter_end = tile_iter + iters_per_tile;
        
        int tile_row = tile_idx / tile_n;
        int tile_col = tile_idx % tile_n;

        //Init Accumulator
        /*Simplify*/
        for (int i = ty; i < BLK_M; i += blockDim.y){
            for (int j = tx; j < BLK_N; j += blockDim.x){
                accum[i][j] = 0.0f;
            }
        }

        //perfor the range of MAC iteration for this tile
        int local_iter = iter - tile_iter;
        int local_iter_end = min(iter_end, tile_iter_end) - tile_iter;

        /*============MAC loop================*/
        for (int k_iter = local_iter; k_iter < local_iter_end; k_iter++){
            const int k_base = k_iter * BLK_K;
            for (int i = ty; i < BLK_M; i += blockDim.y){
                const int global_row = tile_row * BLK_M + i;
                if (global_row >= m)
                    continue;

                for (int j = tx; j < BLK_N; j += blockDim.x){
                    const int global_col = tile_col * BLK_N + j;
                    if (global_col >= n)
                        continue;


                    float sum = accum[i][j];
                    for (int kk = 0; kk < BLK_K; kk++){
                        const int global_k = k_base + kk;

                        if (global_k >= k)
                            break;
                        sum += A[global_row * k + global_k] * B[global_k * n + global_col];
                    }
                    accum[i][j] = sum;
                }
            }
        }

        __syncthreads();

        //consolidate partial-sums across CTAs
        bool tile_started = (iter == tile_iter);
        bool tile_ended = (iter_end >= tile_iter_end);
        float* partial_tile = partials + cta_id * BLK_M * BLK_N;
        if(!tile_started){
            /*Store Partials*/
            for(int i = ty; i < BLK_M; i += blockDim.y){
                for(int j = tx; j < BLK_N; j += blockDim.x){
                    partial_tile[i * BLK_N + j] = accum[i][j];
                }
            }

            // Signal completion
            __threadfence();
            if (tx == 0 && ty == 0){
                flags[cta_id] = 1;
            }
        }else{
            if(!tile_ended){
                // accumulate partial sums from other CTA contributing to this tile
                //int cta_end = tile_iter_end/iters_per_tile;
                int last_cta_for_tile = (tile_iter_end - 1) / iter_per_cta;
                for(int cta = cta_id + 1; cta <= last_cta_for_tile; cta++){
                    float* partial_tile = partials + cta * BLK_M * BLK_N;
                    while((volatile int*)flags[cta] == 0){
                        __nanosleep(10);
                    }

                    for(int i = ty; i < BLK_M; i += blockDim.y){
                        for(int j = tx; j < BLK_N; j += blockDim.x){
                            accum[i][j] += partial_tile[i * BLK_N + j];
                        }
                    }
                }
            
            }
            /*Store Tile*/
            for(int i = ty; i < BLK_M; i += blockDim.y){
                for(int j = tx; j < BLK_N; j += blockDim.x){
                    const int global_row = tile_row * BLK_M + i;
                    const int global_col = tile_col * BLK_N + j;
                    if(global_row < m && global_col < n){
                        C[global_row * n + global_col] = accum[i][j];
                    }
                }
            }
        }
        iter = iter_end;
    }
}


int matrix_multiplication (float * A, float * B, float* C,
                            int m, int n, int k, char* methode = "default",
                            float alpha = 0.0, float beta = 1.0){
    /*Mehodes :
        - default
        - sharedM
        - streamK
    */
    float *d_A;
    float *d_B;
    float *d_C;

    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = m * n * sizeof(float);
    
    cudaMalloc((void**)&d_A, bytes_A);
    cudaMalloc((void**)&d_B, bytes_B);
    cudaMalloc((void**)&d_C, bytes_C);


    cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, bytes_C, cudaMemcpyHostToDevice);
    
    // Launch kernel
    //printf("Launching point kernel...\n");
    
    
    int gridDimX = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gridDimY = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gridSize = gridDimX * gridDimY;
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockDimShared(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
    dim3 gridDim(gridDimX, gridDimY);
    
    if (strcmp(methode, "default") == 0){
        point<<<gridDim, blockDim>>>(d_A, d_B, d_C, alpha, beta, m, n, k);
    }else if (strcmp(methode, "sharedM") == 0){
        point_shared<<<gridDim, blockDimShared>>>(d_A, d_B, d_C, m, n, k);
    }else if (strcmp(methode, "streamK") == 0){
        int* flags;
        float * partials;
        
        size_t bytes_flags = gridSize * sizeof(int);
        size_t bytes_partials = gridSize * BLK_M * BLK_N * sizeof(float);
    
        cudaMalloc((void**)&flags, bytes_flags);
        cudaMemset(flags, 0, gridSize * sizeof(int));
        cudaMalloc((void**)&partials, bytes_partials);

        cudaFuncSetAttribute(streamK_point, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
        streamK_point<<<gridDim, blockDim>>>(d_A, d_B, d_C, flags, partials, m, n, k);

        cudaFree(flags);
        cudaFree(partials);
    }else{
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }
    
    //printf("Kernel execution completed!\n");
    
    // Copy result back to host
    //printf("Copying results back to host...\n");
    cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
