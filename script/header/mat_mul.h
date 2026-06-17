#ifndef MAT_MUL_H
#define MAT_MUL_H

__device__ inline int ceil_div(int a, int b); 

int matrix_multiplication (float * A, float * B, float* C,
                            int m, int n, int k, char* methode = "default", 
                            bool perf = false, float alpha = 0.0, float beta = 1.0);
/*General Matrix Multiplication using parallele compluting.
        Compute (alpha * C) + beta * (A@B)

    A = (m, k)  |  B = (k, n)  | C = (m, n)
    Mehodes :
        - default => simple no optimization
        - sharedM => unsing shared memory + transposed B matrix
        - streamK => tile decomposition (NOT Working)
*/


#endif