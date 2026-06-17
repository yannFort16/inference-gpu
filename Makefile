# Makefile for CUDA test programs in inference-gpu

NVCC        := nvcc
NVCC_FLAGS  := -m64 -diag-suppress 2464
SRC_DIR     := script/test
OP_DIR      := script/operation
OUT_DIR     := exec

.PHONY: all clean

all: main_mat_mul.exe tests_mat_mut.exe

main_mat_mul.exe: $(SRC_DIR)/main_mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

tests_mat_mut.exe: $(SRC_DIR)/tests_mat_mut.cu $(OP_DIR)/mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

test_convolution.exe: $(SRC_DIR)/test_convolution.cu $(OP_DIR)/convolution.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f (OP_DIR)/main_mat_mul.exe (OP_DIR)/tests_mat_mut.exe (OP_DIR)/test_convolution.exe
