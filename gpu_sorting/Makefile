# Compiler and flags
CC = gcc
CFLAGS = -c -std=c99 -Wall -Wextra -I.
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -c -rdc=true

# Files
C_FILES = util.c
CUDA_FILES = gpu_util.cu testQuickSort.cu largeMergeSort_with_mergesort.cu

# Objects
C_OBJS = $(C_FILES:.c=.o)
CUDA_OBJS = $(CUDA_FILES:.cu=.o)
OBJS = $(C_OBJS) $(filter-out _with_mergesort.o, $(CUDA_OBJS))
LARGEMERGE_OBJS = $(C_OBJS) gpu_util.o largeMergeSort_with_mergesort.o

# Output
OUTPUT = testQuickSort
MERGESORT_OUTPUT = mergesort

# Rules
all: $(OUTPUT) $(MERGESORT_OUTPUT)

$(OUTPUT): $(OBJS)
	$(NVCC) $(OBJS) -o $@ -lcudart

$(MERGESORT_OUTPUT): $(LARGEMERGE_OBJS)
	$(NVCC) $(LARGEMERGE_OBJS) -o $@ -lcudart

%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(C_OBJS) $(CUDA_OBJS) $(OUTPUT) $(MERGESORT_OUTPUT) merge_error_log.txt mergeoutput.txt

.PHONY: clean
