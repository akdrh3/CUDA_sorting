#include "util.h"

// Merges two subarrays of arr[].
// First subarray is arr[left..mid]
// Second subarray is arr[mid+1..right]
void merge(int arr[], uint64_t left, uint64_t mid, uint64_t right) {
    uint64_t i, j, k;
    uint64_t n1 = mid - left + 1;
    uint64_t n2 = right - mid;

    // Create temporary arrays
    int leftArr[n1], rightArr[n2];

    // Copy data to temporary arrays
    for (i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        rightArr[j] = arr[mid + 1 + j];

    // Merge the temporary arrays back into arr[left..right]
    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        }
        else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of leftArr[], if any
    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }

    // Copy the remaining elements of rightArr[], if any
    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
}

// The subarray to be sorted is in the index range [left-right]
void mergeSort(int arr[], uint64_t left, uint64_t right) {
    if (left < right) {
      
        // Calculate the midpoint
        uint64_t mid = left + (right - left) / 2;

        // Sort first and second halves
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // Merge the sorted halves
        merge(arr, left, mid, right);
    }
}

int main() {
    char file_name[256];
    printf("Enter the file name: \n");
    scanf("%255s", file_name);

    uint64_t size_of_array = count_size_of_file(file_name);
    printf("number of array: %lu\n", size_of_array);

    int *arr=NULL;
    read_from_file_cpu(file_name, &arr, size_of_array);

    struct timespec vartime = timer_start();
      // Sorting arr using mergesort
    mergeSort(arr, 0, size_of_array-1);

    long time_elapsed_nanos = timer_end(vartime);
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);

    printf("Merge sort time: %.2f Milliseconds\n", ((float)time_elapsed_nanos)/1000000);
    return 0;
}
