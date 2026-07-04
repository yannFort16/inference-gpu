#ifndef CONV_H
#define CONV_H

float* convolution(float *input, int m, int n, int c, float *filter, int k,
                char* methode  = "default", bool pad = false, int stride = 1, 
                float pad_val = 0, bool perf = false);
/* methode is either : default || shared || split

    input : input data composed of 2D array over the c chanels
   dim(input) = (c, m, n)
   
   filter : filters to be pas over the input
   dim(kernel) = (c, k, k)

   if pad is true, the padding will be composed of pad_val. Works only for k odd.
*/ 


#endif