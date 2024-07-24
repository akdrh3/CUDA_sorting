#include "gpu_util.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
extern "C" {
#include "util.h"
}

int main() {
    // Read the numbers from the file into an array in CPU memory.
    char file_name[256];
    printf("Enter the file name: ");
    scanf("%255s", file_name);

    uint64_t size_of_array = count_size_of_file(file_name);
    printf("Number of integers in the file : %lu\n", size_of_array);

    int *number_array = NULL;
    read_from_file(file_name, &number_array, size_of_array);
    printf("Last element: %d\n", number_array[size_of_array - 1]);

    // Allocate memory on the GPU.

    cudaEvent_t start, stop;
    cuda_timer_start(&start, &stop);

    int *gpu_number_array = NULL;
    HANDLE_ERROR(cudaMalloc(&gpu_number_array, sizeof(int) * size_of_array));
    HANDLE_ERROR(cudaMemcpy(gpu_number_array, number_array, sizeof(int) * size_of_array, cudaMemcpyHostToDevice));

    double time_copy_number_array_to_gpu = cuda_timer_stop(start, stop);
    int last_element;
    HANDLE_ERROR(cudaMemcpy(&last_element, gpu_number_array + size_of_array - 1, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Last element on cudamalloc: %d\n Time elipsed to copy array to gpu: %lf\n", last_element, time_copy_number_array_to_gpu);

    HANDLE_ERROR(cudaFree(gpu_number_array));
    free(number_array);

    return 0;
}

// Copy the array from CPU memory to GPU memory.