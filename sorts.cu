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

// CUDA kernel for quick sort partition
__device__ int partition(int *arr, int low, int high)
{
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++)
    {
        if (arr[j] <= pivot)
        {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return i + 1;
}

// CUDA kernel for quick sort
__global__ void quickSort(int *arr, int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, low, high);

        quickSort<<<1, 1>>>(arr, low, pi - 1);
        quickSort<<<1, 1>>>(arr, pi + 1, high);
        cudaDeviceSynchronize();
    }
}

// Function to perform sorting and measure time
void performSortAndMeasureTime(const std::string &filename)
{
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
    int *d_temp;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Measure merge sort time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mergeSortHost(d_arr, d_temp, 0, n - 1);
    cudaEventRecord(stop);

    cudaMemcpy(vec.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float mergeSortMilliseconds = 0;
    cudaEventElapsedTime(&mergeSortMilliseconds, start, stop);

    std::cout << "Merge sort time for " << filename << ": " << mergeSortMilliseconds << " ms" << std::endl;

    cudaMemcpy(d_arr, vec.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Measure quick sort time
    cudaEventRecord(start);
    quickSort<<<1, 1>>>(d_arr, 0, n - 1);
    cudaEventRecord(stop);

    cudaMemcpy(vec.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float quickSortMilliseconds = 0;
    cudaEventElapsedTime(&quickSortMilliseconds, start, stop);

    std::cout << "Quick sort time for " << filename << ": " << quickSortMilliseconds << " ms" << std::endl;

    cudaFree(d_arr);
    cudaFree(d_temp);
}

int main()
{
    std::vector<std::string> filenames = {
        "oneMillionNum.txt",
        "twoMillionNum.txt",
        "fourMillionNum.txt",
        "eightMillionNum.txt",
        "sxtnMillionNum.txt",
        "thrtytwMillionNum.txt"};

    for (const auto &filename : filenames)
    {
        performSortAndMeasureTime(filename);
    }

    return 0;
}