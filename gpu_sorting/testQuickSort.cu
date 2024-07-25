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
            // printf("swapping %d and %d\n", arr[j], arr[i]);
            swap(&arr[i], &arr[j]);
        }
        // printf("j : %ld, high : %ld, i : %ld, arr[j] : %d, pivot : %d\n", j, high, i, arr[j], pivot);
    }
    // printf("last swapping %d and %d\n", arr[i + 1], arr[high]);
    swap(&arr[i + 1], &arr[high]);
    // printf("current array\n");
    // gpu_print_array(arr, 10);

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
    char file_name[256];
    printf("Enter the file name: ");
    scanf("%255s", file_name);

    uint64_t size_of_array = count_size_of_file(file_name);
    printf("Number of integers in the file : %lu\n", size_of_array);

    int *number_array = NULL;
    read_from_file(file_name, &number_array, size_of_array);
    // printf("Last element: %d\n", number_array[size_of_array - 1]);

    // const int64_t size_of_array = 40;
    // int number_array[size_of_array] = {449, 262, 270, 311, 399, 46, 409, 88, 140, 278, 162, 157, 65, 434, 344, 131, 28, 56, 273, 480, 170, 364, 334, 93, 83, 244, 17, 70, 374, 306, 105, 150, 119, 242, 293, 266, 235, 29, 1, 448};
    printf("Original array: \n");
    print_array(number_array, size_of_array);

    // start timer
    cudaEvent_t start, stop;
    cuda_timer_start(&start, &stop);

    // Allocate memory on the GPU.
    int *gpu_number_array = NULL;
    HANDLE_ERROR(cudaMalloc(&gpu_number_array, sizeof(int) * size_of_array));

    // Copy the array from CPU memory to GPU memory.
    HANDLE_ERROR(cudaMemcpy(gpu_number_array, number_array, sizeof(int) * size_of_array, cudaMemcpyHostToDevice));

    // starting quick sort
    quickSortKernel<<<1, 1>>>(gpu_number_array, 0, size_of_array - 1);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // stop timer
    double gpu_sort_time = cuda_timer_stop(start, stop);

    // writing back
    HANDLE_ERROR(cudaMemcpy(number_array, gpu_number_array, sizeof(int) * size_of_array, cudaMemcpyDeviceToHost));
    printf("Sorted array: \n");
    print_array(number_array, size_of_array);

    printf("Time elipsed to copy array to gpu: %lf\n", gpu_sort_time);

    HANDLE_ERROR(cudaFree(gpu_number_array));
    free(number_array);

    return 0;
}
