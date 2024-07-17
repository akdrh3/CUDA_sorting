#include "cuda_profiler_api.h"
#include "gpu_util.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ int partition(int *arr, int left, int right) {
    int pivot = arr[right];
    int i = left - 1;

    for (int j = left; j <= right - 1; ++j) {
        if (arr[j] <= pivot) {
            ++i;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[right]);
    return (i + 1);
}

__global__ void quickSort(int *arr, int left, int right) {
    // Stack for storing left and right indices
    int stack[1024];

    // Initialize stack
    int top = -1;
    stack[++top] = left;
    stack[++top] = right;

    // Pop from stack and push sub-arrays
    while (top >= 0) {
        right = stack[top--];
        left = stack[top--];

        int p = partition(arr, left, right);

        // If there are elements on the left side of the pivot, push left side to stack
        if (p - 1 > left) {
            stack[++top] = left;
            stack[++top] = p - 1;
        }

        // If there are elements on the right side of the pivot, push right side to stack
        if (p + 1 < right) {
            stack[++top] = p + 1;
            stack[++top] = right;
        }
    }
}

void performQuickSortAndMeasureTime(const std::string &filename) {
    // Read numbers from file
    std::ifstream inputFile(filename);
    if (!inputFile) {
        std::cerr << "Failed to open the file " << filename << std::endl;
        return;
    }

    std::vector<int> vec;
    int number;
    while (inputFile >> number) {
        vec.push_back(number);
    }
    inputFile.close();
    std::cout << "starting quicksort ... " << std::endl;
    uint64_t n = vec.size();
    printf("dataset size : %llu", n);
    int *d_arr;
    printf("Allocating memory on GPU\n");
    HANDLE_ERROR(cudaMallocManaged(&d_arr, n * sizeof(int)));

    // Measure time
    printf("start quick sort ...\n");
    cudaEvent_t start, stop;
    cuda_timer_start(&start, &stop);
    quickSort<<<1, 1>>>(d_arr, 0, n - 1);
    printf("finished quick sort ...\n");
    double milliseconds = cuda_timer_stop(start, stop);

    HANDLE_ERROR(cudaDeviceSynchronize());

    double seconds = milliseconds / 1000.0;

    std::cout << "Time taken to quick sort " << n << " elements: " << seconds << " s" << std::endl;
    printf("Time taken to quick sort %llu elements: %ld s\n", n, seconds);
    HANDLE_ERROR(cudaFree(d_arr));
    // // Optionally, print the sorted array
    // std::cout << "Sorted output: ";
    // for (int i = 0; i < n; ++i)
    // {
    //     std::cout << vec[i] << " ";
    // }
    // std::cout << std::endl;
}

int main() {
    performQuickSortAndMeasureTime("numbers.txt");
    return 0;
}