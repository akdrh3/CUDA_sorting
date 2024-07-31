#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__device__ void quickSortKernel(int *arr, int left, int right)
{
    if (left < right)
    {
        int i = left, j = right;
        int pivot = arr[(left + right) / 2];

        while (i <= j)
        {
            while (arr[i] < pivot)
                i++;
            while (arr[j] > pivot)
                j--;
            if (i <= j)
            {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                i++;
                j--;
            }
        }

        if (left < j)
            quickSortKernel(arr, left, j);
        if (i < right)
            quickSortKernel(arr, i, right);
    }
}

__global__ void quickSort(int *arr, int left, int right)
{
    quickSortKernel(arr, left, right);
}

int main()
{
    std::vector<int> vec = {34, 7, 23, 32, 5, 62};
    int n = vec.size();

    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    quickSort<<<1, 1>>>(d_arr, 0, n - 1);
    cudaDeviceSynchronize();

    cudaMemcpy(vec.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    for (int i = 0; i < n; ++i)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}