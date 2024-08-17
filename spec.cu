#include <iostream>
#include <cuda_runtime.h>

int main()
{
    cudaDeviceProp prop;
    int device;
    
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dim: " 
              << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " 
              << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max Grid Size: " 
              << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " 
              << prop.maxGridSize[2] << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Number of Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

    return 0;
}
