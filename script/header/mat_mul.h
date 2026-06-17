#ifndef MAT_MUL_H
#define MAT_MUL_H

__device__ inline int ceil_div(int a, int b); 

int matrix_multiplication (float * A, float * B, float* C,
                            int m, int n, int k, char* methode = "default", 
                            bool perf = false, float alpha = 0.0, float beta = 1.0);

#endif