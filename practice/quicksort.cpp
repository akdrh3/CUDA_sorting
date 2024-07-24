#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

void swap(int * a, int * b){
    int tmp = 0;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

int swaptest(){
    int a = 5;
    int b = 6;
    printf("Before swap: a -> %d, b -> %d\n", a ,b);
    swap(&a, &b);
    printf("After swap: a -> %d, b -> %d\n", a ,b);
    return 0;
}

void print_array(int *int_array, int64_t array_size){
    for (int64_t i = 0; i < array_size; ++i){
        printf("%d ", int_array[i]);
    }
    printf("\n");
}

int64_t partition(int arr[], int64_t low, int64_t high){
    int pivot = arr[high];
    int64_t i = low - 1;

    for (int64_t j = low; j<high; j++){
        if(arr[j] < pivot){
            i = i + 1;
            printf("swapping %d and %d\n", arr[j], arr[i]);
            swap(&arr[i], &arr[j]);
        }
        printf("j : %ld, high : %ld, i : %ld, arr[j] : %d, pivot : %d\n", j, high, i, arr[j], pivot);
    }
    //printf("j : %lld, high : %lld, i : %lld, arr[j] : %d, pivot : %d", j, high, i, arr[j], pivot);
    swap(&arr[i+1], &arr[high]);

    return i+1;

}

void quickSort(int arr[], int64_t low, int64_t high){
    if(low <high){
        int64_t pi = partition(arr,low,high);
        quickSort(arr,low,pi-1);
        quickSort(arr,pi+1, high);
    }
}


int main(){
    int64_t array_size = 30;
    int numbers[array_size] = {7, 8, 10, 12, 14, 16, 18, 19, 20, 26, 36, 41, 44, 47, 51, 53, 56, 61, 65, 67, 69, 70, 71, 72, 77, 78, 81, 83, 85, 100};

    print_array(numbers, array_size);
    quickSort(numbers,0,array_size-1);
    print_array(numbers, array_size);
    
    return 0;
}