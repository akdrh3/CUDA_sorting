#include <iostream>
#include <fstream>
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

void performQuickSortAndMeasureTime(const std::string &filename)
{
    // Read numbers from file
    std::ifstream inputFile(filename);
    if (!inputFile)
    {
        std::cerr << "Failed to open the file " << filename << std::endl;
        return;
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
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    quickSort<<<1, 1>>>(d_arr, 0, n - 1);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    cudaMemcpy(vec.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Quick sort time for " << filename << ": " << milliseconds << " ms" << std::endl;

    // Optionally, print the sorted array
    std::cout << "Sorted output: ";
    for (int i = 0; i < n; ++i)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

int main()
{
    performQuickSortAndMeasureTime("numbers.txt");
    return 0;
}