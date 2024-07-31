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

    int64_t index_of_left_array = 0, index_of_right_array = 0;
    int64_t index_of_merged_array = left;
    // Merge the temp arrays back into arr
    while (index_of_left_array < left_array_size && index_of_right_array < right_array_size) {
        if (left_array[index_of_left_array] <= right_array[index_of_right_array]) {
            arr[index_of_merged_array] = left_array[index_of_left_array];
            index_of_left_array++;
        } else {
            arr[index_of_merged_array] = right_array[index_of_right_array];
            index_of_right_array++;
        }
        index_of_merged_array++;
    }

    // copy the remaining element
    while (index_of_left_array < left_array_size) {
        arr[index_of_merged_array] = left_array[index_of_left_array];
        index_of_left_array++;
        index_of_merged_array++;
    }

    while (index_of_right_array < right_array_size) {
        arr[index_of_merged_array] = right_array[index_of_right_array];
        index_of_right_array++;
        index_of_merged_array++;
    }

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
    print_array(arr, 7);
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

    print_array(number_array, size_of_array);

    return 0;
}