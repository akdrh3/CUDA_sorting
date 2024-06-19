#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel to merge two sorted subarrays
__global__ void merge(int *arr, int *temp, int left, int mid, int right)
{
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right)
    {
        if (arr[i] <= arr[j])
        {
            temp[k++] = arr[i++];
        }
        else
        {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid)
    {
        temp[k++] = arr[i++];
    }

    while (j <= right)
    {
        temp[k++] = arr[j++];
    }

    for (int i = left; i <= right; i++)
    {
        arr[i] = temp[i];
    }
}

// Function to recursively split and merge the array on the host
void mergeSortHost(int *d_arr, int *d_temp, int left, int right)
{
    if (left < right)
    {
        int mid = (left + right) / 2;

        // Recursively sort the left and right halves
        mergeSortHost(d_arr, d_temp, left, mid);
        mergeSortHost(d_arr, d_temp, mid + 1, right);

        // Merge the sorted halves on the device
        merge<<<1, 1>>>(d_arr, d_temp, left, mid, right);
        cudaDeviceSynchronize();
    }
}

int main()
{
    std::ifstream inputFile("numbers.txt");
    if (!inputFile)
    {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    std::vector<int> vec;
    int number;
    while (inputFile >> number)
    {
        vec.push_back(number);
    }
    inputFile.close();

    int n = vec.size();
    int *d_arr;
    int *d_temp;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mergeSortHost(d_arr, d_temp, 0, n - 1);
    cudaEventRecord(stop);

    cudaMemcpy(vec.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time taken to merge sort " << n << " elements: " << milliseconds << " ms" << std::endl;

    // Optionally, print the sorted array
    // for (int i = 0; i < n; ++i) {
    //     std::cout << vec[i] << " ";
    // }
    // std::cout << std::endl;

    return 0;
}