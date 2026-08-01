# Makefile for CUDA test programs in inference-gpu

NVCC        := nvcc
NVCC_FLAGS  := -gencode arch=compute_86,code=sm_86 -diag-suppress 2464
SRC_DIR     := script/test
OP_DIR      := script/operation
OUT_DIR     := exec
UTILS		:= script/test/utils.cu
LDFLAGS = -L"$(CUDA_PATH)/lib/x64"
CUDA_PATH	:= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2
.PHONY: all clean

all: main_mat_mul.exe test_mat_mul.exe test_convolution.exe test_pool.exe test_main.exe main.exe

mat_mul: main_mat_mul.exe test_mat_mul.exe

convolution: test_convolution.exe test_pool.exe

main_mat_mul.exe: $(SRC_DIR)/main_mat_mul.cu $(OP_DIR)/mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIR)/$@ $^ $(UTILS) $(LDFLAGS) -lcublas

test_mat_mul.exe: $(SRC_DIR)/test_mat_mul.cu $(OP_DIR)/mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIR)/$@ $^ $(UTILS) $(LDFLAGS) -lcublas

test_convolution.exe: $(SRC_DIR)/test_convolution.cu $(OP_DIR)/convolution.cu $(OP_DIR)/mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIR)/$@ $^ $(UTILS) $(LDFLAGS) -lcublas

test_pool.exe: $(SRC_DIR)/test_pool.cu $(OP_DIR)/pooling.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIR)/$@ $^ $(UTILS) $(LDFLAGS) -lcublas

test_main.exe: $(SRC_DIR)/test_main.cu $(OP_DIR)/mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIR)/$@ $^ $(UTILS) $(LDFLAGS) -lcublas

main.exe: script/main.cu $(OP_DIR)/pooling.cu $(OP_DIR)/convolution.cu $(OP_DIR)/mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -o $(OUT_DIR)/$@ $^ $(UTILS) $(LDFLAGS) -lcublas

clean:
	rm -f $(OUT_DIR)/main_mat_mul.exe $(OUT_DIR)/test_mat_mul.exe $(OUT_DIR)/test_convolution.exe $(OUT_DIR)/test_pool.exe $(OUT_DIR)/test_main.exe  $(OUT_DIR)/main.exe
