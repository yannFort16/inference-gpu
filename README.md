# inference-gpu

CUDA implementations of common neural-network building blocks for GPU execution, including matrix multiplication, convolution, and pooling.

## What is in this project?

This repository contains:

- CUDA matrix multiplication kernels in `script/operation/mat_mul.cu`
- CUDA convolution kernels in `script/operation/convolution.cu`
- CUDA pooling kernels in `script/operation/pooling.cu`
- Small demo programs and validation/benchmark drivers in `script/test/`
- A Makefile that builds everything into the `exec/` folder

The core operation is:

$$C = \alpha \cdot C + \beta \cdot (A \times B)$$

for matrix multiplication, while convolution and pooling are exposed through dedicated GPU kernels.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit with `nvcc`
- A working `make` installation

On Windows, the Makefile currently points `CUDA_PATH` to a specific CUDA installation directory. If your CUDA toolkit is installed elsewhere, update the `CUDA_PATH` line in the Makefile before building.

## Build targets

From the repository root, the available Makefile targets are:

```bash
make all
```

Builds all executables:

- `exec/main_mat_mul.exe`
- `exec/test_mat_mul.exe`
- `exec/test_convolution.exe`
- `exec/test_pool.exe`
- `exec/test_main.exe`
- `exec/main.exe`

Other useful targets:

```bash
make mat_mul
```

Builds the matrix multiplication demos and validation executables.

```bash
make convolution
```

Builds the convolution and pooling demos.

```bash
make clean
```

Removes the generated binaries from `exec/`.

## Running the demos

### 1. Matrix multiplication demo

```bash
./exec/main_mat_mul.exe sharedM
```

This runs a small sample program that:

- creates random matrices `A` and `B`
- runs a matrix multiplication using the requested method
- prints the resulting matrix sample

Supported methods for this demo are the ones exposed by the kernel interface, such as `default`, `sharedM`, `streamK`, and `cuBLAS` depending on the driver.

### 2. General CLI entry point

```bash
./exec/main.exe multiplication <input_A> <input_B> <output> <M> <N> <K> <alpha> <beta> -g
```

This generic driver supports:

- `multiplication`
- `convolution`
- `pooling`

Examples:

```bash
./exec/main.exe multiplication A.txt B.txt C.txt 64 32 16 1.0 0.0 -g
./exec/main.exe convolution input.txt filter.txt out.txt 128 128 3 1 1 true 0.0 -g
./exec/main.exe pooling input.txt filter.txt out.txt 128 128 3 1 1 true 0.0 -g
```

The `-g` flag generates random input matrices. Without it, the program reads the input matrices from files.

## Running the tests

### Matrix multiplication validation

```bash
./exec/test_mat_mul.exe default sharedM t
```

This runs a set of validation cases comparing two implementations of matrix multiplication and reports whether they agree.

- `t` stands for test mode
- The program uses several fixed-size test cases and checks correctness

### Convolution validation

```bash
./exec/test_convolution.exe default shared t
```

This validates convolution correctness by comparing two convolution implementations on several test cases.

### Main matrix multiplication verification

```bash
./exec/test_main.exe <input_A> <input_B> <expected_output> <M> <N> <K>
```

This utility reads matrices from files, runs a reference multiplication, and checks the result against the expected output file.

## Running benchmarks

### Matrix multiplication benchmark

```bash
./exec/test_mat_mul.exe default sharedM b
```

This runs a benchmark loop for matrix multiplication using larger problem sizes and prints timing information.

- `b` stands for benchmark mode

### Convolution benchmark

```bash
./exec/test_convolution.exe default shared b
```

This runs a convolution benchmark loop and prints timing information for the two implementations.

## Pooling demo

```bash
./exec/test_pool.exe
```

This launches a small pooling example with a random input matrix and prints the pooling result.

## Notes

- The source code and headers live under `script/operation/` and `script/header/`.
- The matrix multiplication kernel supports modes such as `default`, `sharedM`, and `streamK`.
- The convolution interface supports a configurable padding mode and stride.
- The pooling interface supports max or average pooling.
