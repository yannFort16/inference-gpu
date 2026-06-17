# Makefile for CUDA test programs in inference-gpu

NVCC        := nvcc
NVCC_FLAGS  := -m64 -diag-suppress 2464
SRC_DIR     := script/test
OP_DIR      := script/operation
OUT_DIR     := exec
UTILS		:= script/test/utils.cu
.PHONY: all clean

all: main_mat_mul.exe tests_mat_mut.exe test_convolution.exe

main_mat_mul.exe: $(SRC_DIR)/main_mat_mul.cu $(OP_DIR)/mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIR)/$@ $^ $(UTILS)

tests_mat_mut.exe: $(SRC_DIR)/tests_mat_mut.cu $(OP_DIR)/mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIR)/$@ $^ $(UTILS)

test_convolution.exe: $(SRC_DIR)/test_convolution.cu $(OP_DIR)/convolution.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIR)/$@ $^ $(UTILS)

clean:
	rm -f $(OUT_DIR)/main_mat_mul.exe $(OUT_DIR)/tests_mat_mut.exe $(OUT_DIR)/test_convolution.exe
