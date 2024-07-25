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

__device__ int64_t partition(int arr[], int64_t low, int64_t high) {
    int pivot = arr[high];
    int64_t i = low - 1;

    for (int64_t j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i = i + 1;
            // printf("swapping %d and %d\n", arr[j], arr[i]);
            swap(&arr[i], &arr[j]);
        }
        // printf("j : %ld, high : %ld, i : %ld, arr[j] : %d, pivot : %d\n", j, high, i, arr[j], pivot);
    }
    // printf("j : %lld, high : %lld, i : %lld, arr[j] : %d, pivot : %d", j, high, i, arr[j], pivot);
    swap(&arr[i + 1], &arr[high]);

    return i + 1;
}

__global__ void quickSortKernel(int *arr, int64_t low, int64_t high) {
    if (low < high) {
        int64_t pi = partition(arr, low, high);
        quickSortKernel<<<1, 1>>>(arr, low, pi - 1);
        quickSortKernel<<<1, 1>>>(arr, pi + 1, high);
    }
}

void print_array(int *int_array, int64_t array_size) {
    for (int64_t i = 0; i < array_size; ++i) {
        printf("%d ", int_array[i]);
    }
    printf("\n");
}

int main() {
    // // Read the numbers from the file into an array in CPU memory.
    // char file_name[256];
    // printf("Enter the file name: ");
    // scanf("%255s", file_name);

    // uint64_t size_of_array = count_size_of_file(file_name);
    // printf("Number of integers in the file : %lu\n", size_of_array);

    // int *number_array = NULL;
    // read_from_file(file_name, &number_array, size_of_array);
    // printf("Last element: %d\n", number_array[size_of_array - 1]);

    const int64_t size_of_array = 30;
    int number_array[size_of_array] = {7, 8, 10, 12, 14, 16, 18, 19, 20, 26, 36, 41, 44, 47, 51, 53, 56, 61, 65, 67, 69, 70, 71, 72, 77, 78, 81, 83, 85, 100};
    printf("Original array: \n");
    print_array(number_array, size_of_array);

    cudaEvent_t start, stop;
    cuda_timer_start(&start, &stop);

    // Allocate memory on the GPU.
    int *gpu_number_array = NULL;
    HANDLE_ERROR(cudaMalloc(&gpu_number_array, sizeof(int) * size_of_array));

    // Copy the array from CPU memory to GPU memory.
    HANDLE_ERROR(cudaMemcpy(gpu_number_array, number_array, sizeof(int) * size_of_array, cudaMemcpyHostToDevice));

    quickSortKernel<<<1, 1>>>(gpu_number_array, 0, size_of_array - 1);
    double gpu_sort_time = cuda_timer_stop(start, stop);
    int last_element;
    HANDLE_ERROR(cudaMemcpy(number_array, gpu_number_array, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Sorted array: \n");
    print_array(number_array, size_of_array);

    printf("Time elipsed to copy array to gpu: %lf\n", gpu_sort_time);

    HANDLE_ERROR(cudaFree(gpu_number_array));
    free(number_array);

    return 0;
}
