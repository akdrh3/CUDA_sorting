#include "gpu_util.cuh"
extern "C" {
#include "util.h"
}

void print_array(int *int_array, int64_t array_size) {
    for (int64_t i = 0; i < array_size; ++i) {
        printf("%d ", int_array[i]);
    }
    printf("\n");
}

__device__ void merge(int* arr, int* tmp, int left, int mid, int right)
{
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right)
    {
        if (arr[i] <= arr[j])
        {
            tmp[k++] = arr[i++];
        }
        else
        {
            tmp[k++] = arr[j++];
        }
    }

    while (i <= mid)
    {
        tmp[k++] = arr[i++];
    }

    while (j <= right)
    {
        tmp[k++] = arr[j++];
    }

    for (i = left; i <= right; i++)
    {
        arr[i] = tmp[i];
    }
}

__global__ void mergeSortKernel(int* arr, int* tmp, int left, int right, int chunkSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int currentChunk = tid * chunkSize;
    int mid = min(currentChunk + chunkSize / 2 - 1, right);
    int end = min(currentChunk + chunkSize - 1, right);

    // Boundary check to avoid illegal memory access
    if (currentChunk >= right || currentChunk < left)
    {
        return;  // Ignore out-of-bounds threads
    }

    if (currentChunk < end)
    {
        merge(arr, tmp, currentChunk, mid, end);
    }
}


void mergesort(int* arr, int* tmp, int left, int right, int threadsPerBlock)
{
    int blockSize = threadsPerBlock;
    int gridSize = (right - left + 1 + blockSize - 1) / blockSize;
    printf("blockSize : %d, gridSize : %d", blockSize, gridSize);

    for (int chunkSize = 2; chunkSize <= right - left + 1; chunkSize *= 2)
    {
        mergeSortKernel<<<gridSize, blockSize>>>(arr, tmp, left, right, chunkSize);
        HANDLE_ERROR(cudaDeviceSynchronize());  // Synchronize after each kernel launch
    }
}


int main() 
{
    // read the numbers from the file into an array in CPU memory.
    char file_name[256];
    printf("Enter the file name: \n");
    scanf("%255s", file_name);

    uint64_t size_of_array = count_size_of_file(file_name);
    printf("Number of integers in the file: %lu\n", size_of_array);

    int *number_array = NULL;
    read_from_file(file_name, &number_array, size_of_array);

    // Array to store the different thread counts
    int threads_options[5] = {1, 256, 512, 768, 1024};

    // Loop through each thread count
    for (int t = 0; t < 5; t++) 
    {
        int threads_per_block = threads_options[t];

        // Allocate memory on GPU for each run
        int *gpu_arr = NULL;
        int *gpu_tmp = NULL;
        HANDLE_ERROR(cudaMallocManaged((void **)&gpu_arr, size_of_array * sizeof(int)));
        HANDLE_ERROR(cudaMallocManaged((void **)&gpu_tmp, size_of_array * sizeof(int)));

        memcpy(gpu_arr, number_array, size_of_array * sizeof(int));

        // Start timer
        cudaEvent_t start, stop;
        cuda_timer_start(&start, &stop);
        printf("\nRunning Merge Sort with %d threads per block . . . \n", threads_per_block);

        // Calculate the number of blocks needed for the merge step
        int blocks = (size_of_array + threads_per_block - 1) / threads_per_block;

        // Run mergesort with the current thread count
        mergesort(gpu_arr, gpu_tmp, 0, size_of_array - 1, threads_per_block);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Stop timer
        double gpu_sort_time = cuda_timer_stop(start, stop);
        double gpu_sort_time_sec = gpu_sort_time / 1000.0;

        printf("Time elapsed for merge sort with %d threads: %lf s\n", threads_per_block, gpu_sort_time_sec);

        HANDLE_ERROR(cudaFree(gpu_arr));
        HANDLE_ERROR(cudaFree(gpu_tmp));
    }

    free(number_array);

    return 0;
}



// __global__ void merge(int *arr, int *tmp, int64_t left, int64_t mid, int64_t right) {
//     int i = left;
//     int j = mid + 1;
//     int k = left;

//     while (i <= mid && j <= right) {
//         if (arr[i] <= arr[j]) {
//             tmp[k++] = arr[i++];
//         } else {
//             tmp[k++] = arr[j++];
//         }
//     }

//     while (i <= mid) {
//         tmp[k++] = arr[i++];
//     }

//     while (j <= right) {
//         tmp[k++] = arr[j++];
//     }

//     for (i = left; i <= right; i++) {
//         arr[i] = tmp[i];
//     }
// }

// void mergesort(int *arr, int *tmp, int64_t const begin, int64_t const end, int thread_per_block) {

//     if (begin >= end) {
//         return;
//     }

//     int64_t mid = begin + (end - begin) / 2;

//     mergesort(arr, tmp, begin, mid, thread_per_block);
//     mergesort(arr, tmp, mid + 1, end, thread_per_block);

//     merge<<<1, 1>>>(arr, tmp, begin, mid, end);
//     cudaDeviceSynchronize();
// }