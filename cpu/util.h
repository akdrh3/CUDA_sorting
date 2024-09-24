#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void read_from_file_cpu(char *file_name, int **numbers, uint64_t size_of_array);
struct timespec timer_start();
long timer_end(struct timespec start_time);