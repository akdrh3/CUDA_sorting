#include <cstdint>
#include <stdio.h>
#include <stdlib.h>

void print_array(int *int_array, int64_t array_size) {
    for (int64_t i = 0; i < array_size; ++i) {
        printf("%d ", int_array[i]);
    }
    printf("\n");
}

void merge(int *arr, int64_t const left, int64_t const mid, int64_t const right){
    int64_t const left_array_size = mid - left + 1;
    int64_t const right_array_size = right - mid;

    printf("merging begin : %lu, mid : %lu, end : %lu, array[b] : %d, array[m] : %d, array[e] : %d\n", left, mid, right, arr[left], arr[mid], arr[right]);

    // int *left_array = (*int)malloc(left_array_size * sizeof(int));
    // int *right_array = (*int)malloc(right_array_size * sizeof(int));

}

void mergesort(int *arr, int64_t const begin, int64_t const end){
    if(begin >= end){
        printf("single element : %lu, array[i] : %d\n", begin, arr[begin]);
        return;
    }

    int64_t mid = begin + (end - begin)/2;
    printf("begin : %lu, mid : %lu, end : %lu, array[b] : %d, array[m] : %d, array[e] : %d\n", begin, mid, end, arr[begin], arr[mid], arr[end]);

    mergesort(arr, begin, mid);
    mergesort(arr, mid+1, end);
    merge(arr, begin, mid, end);
}

int main(){
    const int64_t array_size = 7;
    int numbers[array_size] = {38, 27, 43, 3, 9, 82, 10};

    print_array(numbers, 7);

    printf("mergesort start");
    mergesort(numbers, 0, array_size -1 );

    return 0;
}