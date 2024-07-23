#include "gpu_util.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

void read_from_file(char *file_name, int **numbers, uint64_t size_of_array) {
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    *numbers = (int *)malloc(size_of_array * sizeof(int));
    if (*numbers == NULL) {
        perror("Error allocating memory");
        exit(EXIT_FAILURE);
    }

    for (uint64_t i = 0; i < size_of_array; i++) {
        if (fscanf(file, "%d", &(*numbers)[i]) == EOF) {
            perror("Error reading from file");
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}

uint64_t count_size_of_file(char *file_name) {
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    uint64_t count = 0;
    int tmp = 0;
    while (fscanf(file, "%d", &tmp) != EOF) {
        count++;
    }

    fclose(file);
    return count;
}

int main() {
    char file_name[256];
    printf("Enter the file name: ");
    scanf("%255s", file_name);

    uint64_t size_of_array = count_size_of_file(file_name);
    printf("Number of integers in the file : %llu\n", size_of_array);

    int *number_array = NULL;
    read_from_file(file_name, &number_array, size_of_array);
    printf("Last element: %d\n", number_array[size_of_array - 1]);

    free(number_array);

    return 0;
}