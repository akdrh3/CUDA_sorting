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

// __global__ void old_mergeKernel(int *gpu_arr, int *gpu_left_array, int *gpu_right_array, int64_t left_array_size, int64_t right_array_size) {

//     int64_t index_of_left_array = 0, index_of_right_array = 0;
//     int64_t index_of_merged_array = 0;

//     // Merge the temp arrays back into arr
//     while (index_of_left_array < left_array_size && index_of_right_array < right_array_size) {
//         if (gpu_left_array[index_of_left_array] <= gpu_right_array[index_of_right_array]) {
//             gpu_arr[index_of_merged_array] = gpu_left_array[index_of_left_array];
//             index_of_left_array++;
//         } else {
//             gpu_arr[index_of_merged_array] = gpu_right_array[index_of_right_array];
//             index_of_right_array++;
//         }
//         index_of_merged_array++;
//     }

//     // copy the remaining element
//     while (index_of_left_array < left_array_size) {
//         gpu_arr[index_of_merged_array] = gpu_left_array[index_of_left_array];
//         index_of_left_array++;
//         index_of_merged_array++;
//     }

//     while (index_of_right_array < right_array_size) {
//         gpu_arr[index_of_merged_array] = gpu_right_array[index_of_right_array];
//         index_of_right_array++;
//         index_of_merged_array++;
//     }
// }

// void old_merge(int *arr, int64_t const left, int64_t const mid, int64_t const right) {
//     int64_t const left_array_size = mid - left + 1;
//     int64_t const right_array_size = right - mid;

//     int *gpu_arr, *gpu_left_arry, *gpu_right_arry;

//     HANDLE_ERROR(cudaMalloc((void **)&gpu_arr, (right - left + 1) * sizeof(int)));
//     HANDLE_ERROR(cudaMalloc((void **)&gpu_left_arry, left_array_size * sizeof(int)));
//     HANDLE_ERROR(cudaMalloc((void **)&gpu_right_arry, right_array_size * sizeof(int)));

//     HANDLE_ERROR(cudaMemcpy(gpu_left_arry, arr + left, left_array_size * sizeof(int), cudaMemcpyHostToDevice));
//     HANDLE_ERROR(cudaMemcpy(gpu_right_arry, arr + mid + 1, right_array_size * sizeof(int), cudaMemcpyHostToDevice));

//     old_mergeKernel<<<1, 1>>>(gpu_arr, gpu_left_arry, gpu_right_arry, left_array_size, right_array_size);
//     HANDLE_ERROR(cudaDeviceSynchronize());

//     HANDLE_ERROR(cudaMemcpy(arr + left, gpu_arr, (right - left + 1) * sizeof(int), cudaMemcpyDeviceToHost));
//     cudaFree(gpu_arr);
//     cudaFree(gpu_left_arry);
//     cudaFree(gpu_right_arry);
// }

__global__ void merge(int *arr, int *tmp, int64_t left, int64_t mid, int64_t right) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            tmp[k++] = arr[i++];
        } else {
            tmp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        tmp[k++] = arr[i++];
    }

    while (j <= right) {
        tmp[k++] = arr[j++];
    }

    for (i = left; i <= right; i++) {
        arr[i] = tmp[i];
    }
}

void mergesort(int *arr, int *tmp, int64_t const begin, int64_t const end) {

    if (begin >= end) {
        return;
    }

    int64_t mid = begin + (end - begin) / 2;

    mergesort(arr, tmp, begin, mid);
    mergesort(arr, tmp, mid + 1, end);

    merge<<<1, 1>>>(arr, tmp, begin, mid, end);
    cudaDeviceSynchronize();
}

int main() {

    // read the numbers from the file into an array in CPU memory.
    char file_name[256];
    printf("Enter the file name: \n");
    scanf("%255s", file_name);

    uint64_t size_of_array = count_size_of_file(file_name);
    printf("Number of integers in the file : %lu\n", size_of_array);

    int *number_array = NULL;
    read_from_file(file_name, &number_array, size_of_array);

    printf("initial array : \n");
    print_array(number_array, size_of_array);

    int *gpu_arr = NULL;
    int *gpu_tmp = NULL;

    HANDLE_ERROR(cudaMalloc((void **)&gpu_arr, size_of_array * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&gpu_tmp, size_of_array * sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(gpu_arr, number_array, size_of_array * sizeof(int), cudaMemcpyHostToDevice));

    //  start timer
    cudaEvent_t start, stop;
    cuda_timer_start(&start, &stop);
    printf("Start Merge Sort . . . \n");

    mergesort(gpu_arr, gpu_tmp, 0, size_of_array - 1);
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(number_array, gpu_arr, size_of_array * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Sorted array: \n");
    print_array(number_array, size_of_array);

    // stop timer
    double gpu_sort_time = cuda_timer_stop(start, stop);
    double gpu_sort_time_sec = gpu_sort_time / 1000.0;

    printf("Time elipsed for quick sort: %lf s\n", gpu_sort_time_sec);

    HANDLE_ERROR(cudaFree(gpu_arr));
    HANDLE_ERROR(cudaFree(gpu_tmp));
    // print_array(number_array, size_of_array);d

    // writing back
    free(number_array);

    return 0;
}