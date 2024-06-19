#include <iostream>
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
    std::vector<int> vec = {34, 7, 23, 32, 5, 62};
    int n = vec.size();

    int *d_arr;
    int *d_temp;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    mergeSortHost(d_arr, d_temp, 0, n - 1);

    cudaMemcpy(vec.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);

    for (int i = 0; i < n; ++i)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}