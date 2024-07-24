# Compiler and flags
CC = gcc
CFLAGS = -c -std=c99 -Wall -Wextra -I.
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -c

# Files
C_FILES = util.c
CUDA_FILES = gpu_util.cu testQuickSort.cu

# Objects
C_OBJS = $(C_FILES:.c=.o)
CUDA_OBJS = $(CUDA_FILES:.cu=.o)
OBJS = $(C_OBJS) $(CUDA_OBJS)

# Output
OUTPUT = testQuickSort

# Rules
all: $(OUTPUT)

$(OUTPUT): $(OBJS)
	$(NVCC) $(OBJS) -o $@ -lcudart

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(OBJS) $(OUTPUT)

.PHONY: clean
