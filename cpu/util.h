#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void read_from_file_cpu(char *file_name, int **numbers, uint64_t size_of_array);
uint64_t count_size_of_file(char *file_name);
struct timespec timer_start();
long timer_end(struct timespec start_time);