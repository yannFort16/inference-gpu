# inference-gpu

Optimization of Neural Networks Operations for GPU

## Matrix Multiplication : the base of inference of Neural Networks

This repository includes multiple CUDA matrix multiplication implementation in `script/operations/mat_mul.cu` and a simple test script in `script/test/main_mat_mul.cu`. The program computes **`C = (alpha * C) + beta * (A@B)`**

### Build

From the repository root, use `nvcc` to compile the test program:

```bash
nvcc -m64 -diag-suppress 2464 -o matrix_mult.exe script/test/main_mat_mul.cu
```

### Run

After compiling, run the generated executable:

```bash
./matrix_mult.exe
```

The test program will:

- allocate random matrices `A` and `B`
- initialize `C` to zeros. (`C` Can also be allocated but `alpha` has to be changed)
- call `matrix_multiplication(...)` with the `sharedM` method
- print sample values for input matrices and the resulting output matrix.

### Notes

- The matrix multiplication logic is defined in `script/operations/mat_mul.cu`.
- The `matrix_multiplication` function supports three modes: `default`, `sharedM`, and `streamK`.
- The sample driver currently uses `sharedM` for better GPU memory performance.
