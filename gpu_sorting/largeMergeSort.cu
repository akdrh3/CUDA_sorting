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

void swap_int_pointer(int *arr_A, int *arr_B){
    int *tmp_pointer=NULL;
    tmp_pointer = arr_A;
    arr_A = tmp;
    arr_B = tmp_pointer;
}

__device__ void merge(int* arr, int* tmp, uint64_t start, uint64_t mid, uint64_t end)
{
    uint64_t array_a_index = start, array_b_index = mid, temp_index = start;

    while (array_a_index <= mid && array_b_index <= end){
        if (arr[array_a_index] <= arr[array_b_index]){
            tmp[temp_index++] = arr[array_a_index++];
        } 
        else{
            tmp[temp_index++] = arr[array_b_index++];
        }
    }

    while (array_a_index <= mid){
        tmp[temp_index++] = arr[array_a_index++];
    }

    while (array_b_index <= end){
        tmp[temp_index++] = arr[array_b_index++];
    }
    
}

__global__ void mergeSortKernel(int* arr, int* tmp, uint64_t right, uint64_t chunkSize)
{
    uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint64_t starting_index = tid * chunkSize; //last tid * chunkSize = size_of_array - chunksize
    
    
    uint64_t mid = min(starting_index + chunkSize / 2 , right);
    uint64_t end = min(starting_index + chunkSize - 1, right);

    // Ignore out-of-bounds threads
    if (starting_index >= right){
        return;  
    }

    printf("tid: %lu, chunkSize : %lu, starting index: %lu, mid: %lu, end: %lu, size of array: %lu\n", tid, chunkSize, starting_index, mid, end, right);

    if (starting_index < end){
        merge(arr, tmp, starting_index, mid, end);
    }
}


void mergesort(int* arr, int* tmp, uint64_t size_of_array, uint64_t blockSize)
{
    int gridSize = (size_of_array + blockSize - 1) / blockSize;
    printf("blockSize : %d, gridSize : %d\n", blockSize, gridSize);

    for (uint64_t chunkSize = 2; chunkSize <= size_of_array*2; chunkSize *= 2)
    {
        mergeSortKernel<<<gridSize, blockSize>>>(arr, tmp, size_of_array -1, chunkSize);
        HANDLE_ERROR(cudaDeviceSynchronize());  // Synchronize after each kernel launch
        swap_int_pointer(arr, tmp);
        print_array(arr, size_of_array);
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

    int *gpu_array = NULL;
    int *gpu_tmp = NULL;
    HANDLE_ERROR(cudaMallocManaged((void**)&gpu_array, size_of_array * sizeof(int)));
    HANDLE_ERROR(cudaMallocManaged((void **)&gpu_tmp, size_of_array * sizeof(int)));

    read_from_file(file_name, gpu_array, size_of_array);


    // Array to store the different thread counts
    int threads_options[5] = {1, 256, 512, 768, 1024};

    // Loop through each thread count
    for (int t = 0; t < 5; t++) 
    {
        int threads_per_block = threads_options[t];
        read_from_file(file_name, gpu_array, size_of_array);
  
        // Start timer
        cudaEvent_t start, stop;
        cuda_timer_start(&start, &stop);
        printf("\nRunning Merge Sort with %d threads per block . . . \n", threads_per_block);

        // Calculate the number of blocks needed for the merge step
        uint64_t blocks = (size_of_array + threads_per_block - 1) / threads_per_block;

        // Run mergesort with the current thread count
        mergesort(gpu_array, gpu_tmp, size_of_array, threads_per_block);
        // HANDLE_ERROR(cudaDeviceSynchronize());

        print_array(gpu_array, size_of_array);
        // Stop timer
        double gpu_sort_time = cuda_timer_stop(start, stop);
        double gpu_sort_time_sec = gpu_sort_time / 1000.0;

        printf("Time elapsed for merge sort with %d threads: %lf s\n", threads_per_block, gpu_sort_time_sec);
        
    }
    HANDLE_ERROR(cudaFree(gpu_tmp));
    HANDLE_ERROR(cudaFree(gpu_array));  
    return 0;
}

