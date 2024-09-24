#include "util.h"

void read_from_file_cpu(char *file_name, int **numbers, uint64_t size_of_array) {
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

struct timespec timer_start(){
    struct timespec start_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    return start_time;
}

long timer_end(struct timespec start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_REALTIME, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
    return diffInNanos;
}