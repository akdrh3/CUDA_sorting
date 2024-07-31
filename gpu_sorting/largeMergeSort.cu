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

__global__ void mergeKernel(int *gpu_arr, int *gpu_left_array, int *gpu_right_array, int64_t left_array_size, int64_t right_array_size, int64_t start_index) {

    int64_t index_of_left_array = 0, index_of_right_array = 0;
    int64_t index_of_merged_array = start_index;

    // Merge the temp arrays back into arr
    while (index_of_left_array < left_array_size && index_of_right_array < right_array_size) {
        if (gpu_left_array[index_of_left_array] <= gpu_right_array[index_of_right_array]) {
            gpu_arr[index_of_merged_array] = gpu_left_array[index_of_left_array];
            index_of_left_array++;
        } else {
            gpu_arr[index_of_merged_array] = gpu_right_array[index_of_right_array];
            index_of_right_array++;
        }
        index_of_merged_array++;
    }

    // copy the remaining element
    while (index_of_left_array < left_array_size) {
        gpu_arr[index_of_merged_array] = gpu_left_array[index_of_left_array];
        index_of_left_array++;
        index_of_merged_array++;
    }

    while (index_of_right_array < right_array_size) {
        gpu_arr[index_of_merged_array] = gpu_right_array[index_of_right_array];
        index_of_right_array++;
        index_of_merged_array++;
    }
}

void merge(int *arr, int64_t const left, int64_t const mid, int64_t const right) {
    int64_t const left_array_size = mid - left + 1;
    int64_t const right_array_size = right - mid;

    // printf("merging begin : %lu, mid : %lu, end : %lu, array[b] : %d, array[m] : %d, array[e] : %d\n", left, mid, right, arr[left], arr[mid], arr[right]);

    int *left_array = (int *)malloc(left_array_size * sizeof(int));
    int *right_array = (int *)malloc(right_array_size * sizeof(int));

    // copy data to temp arrays
    for (int64_t i = 0; i < left_array_size; i++) {
        left_array[i] = arr[left + i];
    }
    for (int64_t j = 0; j < right_array_size; j++) {
        right_array[j] = arr[mid + 1 + j];
    }

    printf("merge begin\n left array: ");
    print_array(left_array, left_array_size);
    printf("right array: ");
    print_array(right_array, right_array_size);

    int *gpu_arr, *gpu_left_arry, *gpu_right_arry;

    HANDLE_ERROR(cudaMalloc((void **)&gpu_arr, (right - left + 1) * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&gpu_left_arry, left_array_size * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&gpu_right_arry, right_array_size * sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(gpu_left_arry, left_array, left_array_size * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpu_right_arry, right_array, right_array_size * sizeof(int), cudaMemcpyHostToDevice));

    mergeKernel<<<1, 1>>>(gpu_arr, gpu_left_arry, gpu_right_arry, left_array_size, right_array_size, left);
    HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaMemcpy(&arr[left], &gpu_arr[left], (right - left + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(gpu_arr);
    cudaFree(gpu_left_arry);
    cudaFree(gpu_right_arry);

    free(left_array);
    free(right_array);
}

void mergesort(int *arr, int64_t const begin, int64_t const end) {
    if (begin >= end) {
        printf("single element : %lu, array[i] : %d\n", begin, arr[begin]);
        return;
    }

    int64_t mid = begin + (end - begin) / 2;
    printf("begin : %lu, mid : %lu, end : %lu, array[b] : %d, array[m] : %d, array[e] : %d\n", begin, mid, end, arr[begin], arr[mid], arr[end]);

    mergesort(arr, begin, mid);
    mergesort(arr, mid + 1, end);
    merge(arr, begin, mid, end);
    print_array(arr, end - begin + 1);
}

int main() {

    // read the numbers from the file into an array in CPU memory.
    //  const int64_t array_size = 7;
    //  int numbers[array_size] = {38, 27, 43, 3, 9, 82, 10};
    char file_name[256];
    printf("Enter the file name: \n");
    scanf("%255s", file_name);

    uint64_t size_of_array = count_size_of_file(file_name);
    printf("Number of integers in the file : %lu\n", size_of_array);

    int *number_array = NULL;
    read_from_file(file_name, &number_array, size_of_array);

    // print_array(number_array, size_of_array);
    // start timer
    cudaEvent_t start, stop;
    cuda_timer_start(&start, &stop);
    printf("Start Merge Sort . . . \n");

    // starting merge sort
    mergesort(number_array, 0, size_of_array);

    // stop timer
    double gpu_sort_time = cuda_timer_stop(start, stop);
    double gpu_sort_time_sec = gpu_sort_time / 1000.0;

    // writing back

    return 0;
}