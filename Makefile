# Makefile for building testQuickSort with nvcc and gcc

# Compiler flags
NVCC = /usr/local/cuda/bin/nvcc
GCC = gcc
NVCC_FLAGS = -c
GCC_FLAGS = -c -std=c99 -Wall -Wextra -I.

# Source files
CUDA_SRC = testQuickSort.cu gpu_util.cu
UTIL_SRC = util.c

# Object files
CUDA_OBJ = test.o gpu_util.o
UTIL_OBJ = util.o

# Executable file
EXEC = testfile

# Default target
all: $(EXEC)

# Build the CUDA object file
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_SRC) -o $(CUDA_OBJ)

# Build the util object file
$(UTIL_OBJ): $(UTIL_SRC)
	$(GCC) $(GCC_FLAGS) $(UTIL_SRC) -o $(UTIL_OBJ)

# Link the object files to create the executable
$(EXEC): $(CUDA_OBJ) $(UTIL_OBJ)
	$(NVCC) $(CUDA_OBJ) $(UTIL_OBJ) -o $(EXEC)

# Clean up object files and executable
clean:
	rm -f $(CUDA_OBJ) $(UTIL_OBJ) $(EXEC)

# Phony targets
.PHONY: all clean