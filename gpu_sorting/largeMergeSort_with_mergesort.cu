#include "gpu_util.cuh"
#include <math.h>
extern "C" {
#include "util.h"
}

__device__ void print_array_gpu(int *int_array, int64_t array_size) {
    for (int64_t i = 0; i < array_size; ++i) {
        printf("%d ", int_array[i]);
    }
    printf("\n");
}

__device__ void swap_int_pointer_gpu(int **arr_A, int **arr_B){
    //printf("swapping\n");
    int *tmp_pointer=*arr_A;
    *arr_A = *arr_B;
    *arr_B = tmp_pointer;
    //printf("swapped pointer \n\n");
}

void print_array(int *int_array, int64_t array_size) {
    for (int64_t i = 0; i < array_size; ++i) {
        printf("%d ", int_array[i]);
    }
    printf("\n");
}

void swap_int_pointer(int **arr_A, int **arr_B){
    //printf("swapping\n");
    int *tmp_pointer=*arr_A;
    *arr_A = *arr_B;
    *arr_B = tmp_pointer;
    //printf("swapped pointer \n\n");
}

uint64_t my_ceil(double num) {
    int int_part = (int)num;  // Get the integer part of the number

    // If the number is positive and has a fractional part, increment the integer part
    if (num > int_part) {
        return int_part + 1;
    } else {
        return int_part;
    }
}

__device__ void merge(int* arr, int* tmp, uint64_t start, uint64_t mid, uint64_t end)
{
    uint64_t array_a_index = start, array_b_index = mid, temp_index = start;
    //printf("inside merge; index1 : %lu, index2 : %lu, tmp index: %lu, end: %lu\n", start, mid, start, end);
    while (array_a_index < mid && array_b_index <= end){
        if (arr[array_a_index] <= arr[array_b_index]){
            tmp[temp_index++] = arr[array_a_index++];
        } 
        else{
            tmp[temp_index++] = arr[array_b_index++];
        }
    }

    while (array_a_index < mid){
        tmp[temp_index++] = arr[array_a_index++];
    }

    while (array_b_index <= end){
        tmp[temp_index++] = arr[array_b_index++];
    }  
}

__device__ void initial_merge(int* arr, int* tmp, uint64_t start, uint64_t mid, uint64_t end)
{
    uint64_t array_a_index = start, array_b_index = mid, temp_index = start;
    //printf("inside merge; index1 : %lu, index2 : %lu, tmp index: %lu, end: %lu\n", start, mid, start, end);
    while (array_a_index < mid && array_b_index <= end){
        if (arr[array_a_index] <= arr[array_b_index]){
            tmp[temp_index++] = arr[array_a_index++];
        } 
        else{
            tmp[temp_index++] = arr[array_b_index++];
        }
    }

    while (array_a_index < mid){
        tmp[temp_index++] = arr[array_a_index++];
    }

    while (array_b_index <= end){
        tmp[temp_index++] = arr[array_b_index++];
    }  

        // Now copy the sorted elements from tmp back to the original array 'arr'
    for (uint64_t i = start; i <= end; i++) {
        arr[i] = tmp[i];
    }

}

__global__ void mergeSortKernel(int* arr, int* tmp, uint64_t size_of_array, uint64_t chunkSize, uint64_t blockSize, uint64_t initial_chunk_size)
{
    //getting tid, start, mid, and end index
    uint64_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    // based on tide, devide and get the portion of the array that this specific tid has to work on
    uint64_t starting_index = tid * chunkSize; //last tid * chunkSize = size_of_array - chunksize
    uint64_t mid = min(starting_index + chunkSize / 2 , size_of_array -1);
    uint64_t end = min(starting_index + chunkSize - 1, size_of_array -1);


    // Ignore out-of-bounds threads
    if (starting_index > size_of_array -1){
        return;  
    }


    //check if this is the initial mergesort, which means it needs mergesort inside the kernel
    if (chunkSize == initial_chunk_size){
        //printf("initial mergesort happening inside thread\n");
        //printf("tid: %lu, chunkSize : %lu, blockSize : %lu, starting index: %lu, mid: %lu, end: %lu, size of array: %lu\n", tid, chunkSize, blockSize, starting_index, mid, end, size_of_array);
        uint64_t curr_size, left_start;
        for (curr_size = 1; curr_size <= end; curr_size *= 2){
            for(left_start = starting_index; left_start <= end; left_start += 2*curr_size){
                uint64_t subarray_middle_index = ((left_start + curr_size -1 ) < (end)) ? (left_start + curr_size ) : (end);
                uint64_t right_end = ((left_start + 2*curr_size -1) < (end)) ? (left_start + 2*curr_size -1) : (end);
                if(subarray_middle_index <= right_end){
                    //printf("tid: %lu, curr_size : %lu, left_start : %lu, sub_mid: %lu, right_end: %lu\n", tid, curr_size, left_start, subarray_middle_index, right_end);
                    initial_merge(arr, tmp, left_start, subarray_middle_index, right_end);

                    // printf("gpu_array: ");
                    // print_array_gpu(arr, size_of_array);
                    // printf("gpu_tmp  : ");
                    // print_array_gpu(tmp, size_of_array);
                }
            }
        }
        return;
    }
    //printf("mergesort happening between thread\n");
    //printf("tid: %lu, chunkSize : %lu, blockSize : %lu, starting index: %lu, mid: %lu, end: %lu, size of array: %lu\n", tid, chunkSize, blockSize, starting_index, mid, end, size_of_array);
        


    if (starting_index < end){
        merge(arr, tmp, starting_index, mid, end);
    }
}


void mergesort(int* arr, int* tmp, uint64_t size_of_array, uint64_t blockSize, int*gpu_array, uint64_t initial_chunk_size)
{
    // int gridSize = (size_of_array + blockSize - 1) / blockSize;
    int gridSize = 1;
    //printf("blockSize : %d, gridSize : %d\n", blockSize, gridSize);
    //printf("started mergesort; gpu_array : %p, arr : %p, temp: %p\n", gpu_array, arr, tmp);


    for (uint64_t chunkSize = initial_chunk_size; chunkSize <= size_of_array*2; chunkSize *= 2)
    {
        mergeSortKernel<<<gridSize, blockSize>>>(arr, tmp, size_of_array, chunkSize, blockSize, initial_chunk_size);
        HANDLE_ERROR(cudaDeviceSynchronize());  // Synchronize after each kernel launch
        swap_int_pointer(&arr, &tmp);
        //printf("gpu_array : %p, arr : %p, temp: %p\n", gpu_array, arr, tmp);

        // printf("gpu_array: ");
        // print_array(arr, size_of_array);
        // printf("gpu_tmp  : ");
        // print_array(tmp, size_of_array);
    }
    // Ensure that gpu_array points to the sorted array
    if (arr != gpu_array) {
        //printf("ensure that gpu_array points to the sorted array\n");
        swap_int_pointer(&arr, &tmp);  // Make sure the final sorted array is in arr
        //printf("gpu_array : %p, arr : %p, temp: %p\n", gpu_array, arr, tmp);
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

    //read_from_file(file_name, gpu_array, size_of_array);


    // Array to store the different thread counts
    uint64_t threads_options[5] = {1, 256, 512, 768, 1024};
    cudaEvent_t start, stop;
    // Loop through each thread count
    for (int t = 0; t < 5; t++) 
    {
        uint64_t threads_per_block = threads_options[t];
        read_from_file(file_name, gpu_array, size_of_array);
        uint64_t initial_chunk_size = (uint64_t)ceil((double)size_of_array / threads_per_block);
        printf("initial_chunk_size: %lu\n",initial_chunk_size);

        // Start timer     
        cuda_timer_start(&start, &stop);
        printf("Running Merge Sort with %lu threads per block . . . \n", threads_per_block);

        // Run mergesort with the current thread count
        mergesort(gpu_array, gpu_tmp, size_of_array, threads_per_block, gpu_array, initial_chunk_size);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // Stop timer
        double gpu_sort_time = cuda_timer_stop(start, stop);
        double gpu_sort_time_sec = gpu_sort_time / 1000.0;

        printf("Time elapsed for merge sort with %lu threads: %lf s\n\n", threads_per_block, gpu_sort_time_sec);
        
    }
    HANDLE_ERROR(cudaFree(gpu_tmp));
    HANDLE_ERROR(cudaFree(gpu_array));  
    return 0;
}
