#include "gpu_util.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
extern "C" {
#include "util.h"
}

__device__ void swap(int *a, int *b) {
    int tmp = 0;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

__device__ void gpu_print_array(int *int_array, int64_t array_size) {
    for (int64_t i = 0; i < array_size; ++i) {
        printf("%d ", int_array[i]);
    }
    printf("\n");
}

__device__ int64_t partition(int *arr, int64_t low, int64_t high) {
    int pivot = arr[high];
    int64_t i = low - 1;

    for (int64_t j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i = i + 1;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);

    return i + 1;
}

__global__ void quickSortKernel(int *arr, int64_t size) {
    // Dynamically allocate stack based on the size of the array using shared memory
    extern __shared__ int64_t stack[];
    int64_t top = -1;

    // Push initial low and high indexes
    top++;
    stack[top] = 0;
    top++;
    stack[top] = size - 1;

    // Keep popping from the stack while it's not empty
    while (top >= 0) {
        int64_t high = stack[top];
        top--;
        int64_t low = stack[top];
        top--;

        // Partition the array
        int64_t pi = partition(arr, low, high);

        // Push left side to stack if needed
        if (pi - 1 > low) {
            top++;
            stack[top] = low;
            top++;
            stack[top] = pi - 1;
        }

        // Push right side to stack if needed
        if (pi + 1 < high) {
            top++;
            stack[top] = pi + 1;
            top++;
            stack[top] = high;
        }
    }
}

int main() {
    // Read the numbers from the file into an array in CPU memory.
    char file_name[256];
    printf("Enter the file name: \n");
    scanf("%255s", file_name);

    uint64_t size_of_array = count_size_of_file(file_name);
    printf("Number of integers in the file : %lu\n", size_of_array);

    int *number_array = NULL;
    read_from_file(file_name, &number_array, size_of_array);

    // Allocate memory on the GPU.
    int *gpu_number_array = NULL;
    HANDLE_ERROR(cudaMallocManaged(&gpu_number_array, sizeof(int) * size_of_array));

    // Copy the array from CPU memory to GPU memory.
    memcpy(gpu_number_array, number_array, sizeof(int) * size_of_array);

    // Thread options array
    int threads_options[5] = {1, 256, 512, 768, 1024};

    // Iterate through each thread configuration
    for (int i = 0; i < 5; ++i) {
        // Re-allocate memory on the GPU for each iteration
        HANDLE_ERROR(cudaMallocManaged(&gpu_number_array, sizeof(int) * size_of_array));

        // Copy the array from CPU memory to GPU memory.
        memcpy(gpu_number_array, number_array, sizeof(int) * size_of_array);

        int threadsPerBlock = threads_options[i];
        int blocksPerGrid = (size_of_array + threadsPerBlock - 1) / threadsPerBlock;

        // Print current configuration
        printf("Running QuickSort with %d threads per block...\n", threadsPerBlock);

        // Start timer
        cudaEvent_t start, stop;
        cuda_timer_start(&start, &stop);

        // Launch kernel with different thread configurations
        quickSortKernel<<<blocksPerGrid, threadsPerBlock, sizeof(int64_t) * size_of_array>>>(gpu_number_array, size_of_array);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Stop timer
        double gpu_sort_time = cuda_timer_stop(start, stop);
        double gpu_sort_time_sec = gpu_sort_time / 1000.0;

        // Print elapsed time for the current configuration
        printf("Time elapsed for %d threads per block: %lf s\n\n", threadsPerBlock, gpu_sort_time_sec);

        // Optionally print sorted array
        print_array(gpu_number_array, size_of_array);

        // Free GPU memory for this iteration
        HANDLE_ERROR(cudaFree(gpu_number_array));
    }

    // Free the host memory
    free(number_array);

    return 0;
}
