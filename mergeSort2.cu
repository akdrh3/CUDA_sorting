#include "gpu_util.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

// CUDA kernel to merge two sorted subarrays
__global__ void merge(int *arr, int *temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (int i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

// Function to recursively split and merge the array on the host
void mergeSortHost(int *d_arr, int *d_temp, int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;

        // Recursively sort the left and right halves
        mergeSortHost(d_arr, d_temp, left, mid);
        mergeSortHost(d_arr, d_temp, mid + 1, right);

        // Merge the sorted halves on the device
        merge<<<1, 1>>>(d_arr, d_temp, left, mid, right);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

__host__ void HandleError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void cuda_timer_start(cudaEvent_t *start, cudaEvent_t *stop) {
    HANDLE_ERROR(cudaEventCreate(start));
    HANDLE_ERROR(cudaEventCreate(stop));
    HANDLE_ERROR(cudaEventRecord(*start, 0));
}

__host__ float cuda_timer_stop(cudaEvent_t start, cudaEvent_t stop) {
    float time_elpased;
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time_elpased, start, stop));
    return time_elpased;
}

int main() {
    std::cout << "starting mergesort " << std::endl;
    std::ifstream inputFile("numbers.txt");
    if (!inputFile) {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    std::vector<int> vec;
    int number;
    while (inputFile >> number) {
        vec.push_back(number);
    }
    inputFile.close();

    uint64_t n = vec.size();
    int *d_arr;
    int *d_temp;
    HANDLE_ERROR(cudaMallocManaged(&d_arr, n * sizeof(int)));
    HANDLE_ERROR(cudaMallocManaged(&d_temp, n * sizeof(int)));

    for (uint64_t i = 0; i < n; ++i) {
        d_arr[i] = vec[i];
    }
    memcpy(d_arr, vec.data(), n * sizeof(int));

    // Measure time
    cudaEvent_t start, stop;
    cuda_timer_start(&start, &stop);

    mergeSortHost(d_arr, d_temp, 0, n - 1);

    HANDLE_ERROR(cudaEventSynchronize(stop));
    double milliseconds = cuda_timer_stop(start, stop);
    float seconds = milliseconds / 1000.0;

    std::cout << "Time taken to merge sort " << n << " elements: " << seconds << " s" << std::endl;
    HANDLE_ERROR(cudaFree(d_arr));
    HANDLE_ERROR(cudaFree(d_temp));
    // Optionally, print the sorted array
    // for (int i = 0; i < n; ++i) {
    //     std::cout << vec[i] << " ";
    // }
    // std::cout << std::endl;

    return 0;
}