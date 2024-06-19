#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // make sure we do not go out of bounds
    if (id < n)
    {
        c[id] = a[id] + b[id];
    }
}

int main(int argc, char *argv[])
{
    // size of vectors
    int n = 100000;

    // Host input vectors
    double *h_a;
    double *h_b;
    // host output vectors
    double *h_c;

    // Device input vectors
    double *d_a;
    double *d_b;
    // Device output vectors;
    double *d_c;

    // Size, in byes, of each vec
    size_t bytes = n * sizeof(double);

    // allocate memory for each vector on host
    h_a = (double *)malloc(bytes);
    h_b = (double *)malloc(bytes);
    h_c = (double *)malloc(bytes);

    // allocate memory for each vector on device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // initialize vector on host
    for (i = 0; i < n; i++)
    {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    // copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // num of threads in each thread block
    blockSize = 1024;
    // num of thread block in grid
    gridSize = (int)ceil((float)n / blockSize);

    // execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sum up Vector c and print result devided by n, this should equal 1 within
    double sum = 0;
    for (i = 0; i < n; i++)
    {
        sum += h_c[i];
    }
    printf("final result: %f\n", sum / n);

    // release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}