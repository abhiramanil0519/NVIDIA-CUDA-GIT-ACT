#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define THREADS 128
#define BLOCKS 8
#define MEMSIZE (1<<20)

__global__ void schedulerDemo(int *buffer)
{
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = threadIdx.x / 32;

    if(globalId < MEMSIZE)
        buffer[globalId] = globalId;

    printf("[SM SIM] Block %d | Warp %d | Thread %d | GlobalID %d\n",
           blockIdx.x, warp, threadIdx.x, globalId);
}

void deviceInfo()
{
    int count;
    cudaGetDeviceCount(&count);

    std::cout << "\n==== GPU DEVICE SCAN ====\n";
    std::cout << "Detected devices: " << count << "\n\n";

    for(int i=0;i<count;i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Global Memory: " << prop.totalGlobalMem/(1024*1024) << " MB\n";
        std::cout << "Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "Max Threads/Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Warp Size: " << prop.warpSize << "\n\n";
    }
}

int main()
{
    std::cout << "===== CUDA SYSTEM DIAGNOSTIC TOOL =====\n";

    deviceInfo();

    std::cout << "\n[ALLOC] Reserving GPU memory...\n";

    int *d_mem;
    cudaMalloc(&d_mem, MEMSIZE*sizeof(int));

    std::cout << "[RUN] Launching scheduler simulation\n";

    auto start = std::chrono::high_resolution_clock::now();

    schedulerDemo<<<BLOCKS, THREADS>>>(d_mem);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;

    std::cout << "\n[PERF] Kernel execution time: " << diff.count() << " sec\n";

    std::cout << "[FREE] Releasing memory\n";
    cudaFree(d_mem);

    std::cout << "===== DIAGNOSTIC COMPLETE =====\n";
}
