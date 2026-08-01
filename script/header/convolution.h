#ifndef CONV_H
#define CONV_H


void convolution(float *input, float *output, int m, int n, int c, float *filter, int k,
                char* methode  = "default", bool pad = false, int stride = 1, 
                float pad_val = 0, bool benchmark = false);
/* methode is either : default || shared || split

    input : input data composed of 2D array over the c chanels
   dim(input) = (c, m, n)
   
   filter : filters to be pas over the input
   dim(kernel) = (c, k, k)

   if pad is true, the padding will be composed of pad_val. Works only for k odd.
*/ 

void get_output_dimensions(int m, int n, int k, int stride, bool pad, int* out_m, int* out_n);
/*
    m : height of the input
    n : width of the input
    k : size of the kernel (k x k)
    stride : stride of the convolution
    pad : if true, the output will be padded with zeros
    out_m : height of the output
    out_n : width of the output
*/

#endif