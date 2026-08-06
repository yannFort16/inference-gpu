#ifndef POOL_H
#define POOL_H

void pooling(float *input, float* output, int m, int n, int channels, int k, int stride = 1, 
                bool padding = false, float pad_val = 0.0, char pooling_type = 'm');
/* 
    pooling_type is either : m for max || a for average

    input : input data composed of 2D array over the c chanels
   dim(input) = (c, m, n)
   
   k : size of the kernel to be pas over the input
   stride : stride of the kernel
   padding : if true, the padding will be composed of pad_val. Works only for k odd.

*/

#endif