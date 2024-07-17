CC = gcc
NVCC = /usr/local/cuda/bin/NVCC
CFLAGS = -Wall -Wextra -I. -std=c++11
MERGE_TARGET = mergesort
QUICK_TARGET = quicksort
MERGE_SRC = mergeSort2.cu
QUICK_SRC = quickSort2.cu

HEADERS = gpu_util.cuh 
MERGE_OBJ = $(MERGE_SRC:.cu=.o)
QUICK_OBJ = $(QUICK_SRC:.cu=.o)

all: $(MERGE_TARGET) $(QUICK_TARGET)

#like the object into the final executable for merge sort
# $(MERGE_TARGET): $(MERGE_OBJ)
# 	$(NVCC) $(CFLAGS) -o $(MERGE_TARGET) $(MERGE_OBJ)

$(QUICK_TARGET): $(QUICK_OBJ)
	$(NVCC) $(CFLAGS) -o $(QUICK_TARGET) $(QUICK_OBJ)

#rule to compile .cu files into .o files
# $(MERGE_OBJ): $(MERGE_SRC) $(HEADERS)
# 	$(NVCC) $(CFLAGS) -c $< -o $@

$(QUICK_OBJ): $(QUICK_SRC) $(HEADERS)
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(MERGE_OBJ) $(QUICK_OBJ) $(MERGE_TARGET) $(QUICK_TARGET)

.PHONY: all clean