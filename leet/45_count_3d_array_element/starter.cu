#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Kernel that counts elements matching P using a grid-stride loop
// and atomic operations for reduction.
__global__ void count_elements_kernel(const int* __restrict__ input, int* __restrict__ output, int total_elements, int P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int local_count = 0;
    
    // Grid-stride loop to handle arbitrary array sizes efficiently
    for (int i = idx; i < total_elements; i += stride) {
        if (input[i] == P) {
            local_count++;
        }
    }
    
    // Accumulate the local thread counts into the global output variable
    if (local_count > 0) {
        atomicAdd(output, local_count);
    }
}

extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {
    // Calculate total number of elements in the 3D array
    int total_elements = N * M * K;
    if (total_elements <= 0) return;

    // Initialize the output memory on the host/device to 0 before launching the kernel
    // (Assuming output is managed or we need to guarantee its initial state)
    cudaMemset(output, 0, sizeof(int));

    // Configure execution parameters
    int threadsPerBlock = 256;
    // Calculate grid size, capping it at a reasonable number to avoid launch overhead
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) {
        blocksPerGrid = 1024; 
    }

    // Launch the kernel
    count_elements_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, total_elements, P);
    
    // Ensure kernel execution finishes
    cudaDeviceSynchronize();
}
