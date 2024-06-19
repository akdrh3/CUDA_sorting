#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel.
__global__ void vecMultiply(float *a, float *b, float *c, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N)
    {
        c[id] = a[id] * b[id];
    }
}

void fill_floats(float *x, int size)
{
    for (int i = 0; i < size; i++)
    {
        x[i] = i;
    }
}

int main(int argc, char *argv[])
{
    // size of vector
    int N = 1024;

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    size_t bytes = N * sizeof(float);

    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c = (float *)malloc(bytes);

    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);

    fill_floats(h_a, N);
    fill_floats(h_b, N);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    vecMultiply<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("Vector C \n");
    for (int i = 0; i < N; i++)
    {
        printf("%f\n", h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}