#include <cuda_runtime.h>

__global__ void gemv_shared_kernel(const float* A, const float* x, float* y, int M, int N) {
    // Dynamically allocated shared memory for the vector x
    extern __shared__ float x_shared[];
    
    // Cooperatively load x into shared memory across all threads in the block
    // If N is larger than the block size (256), threads will loop to load multiple elements
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        x_shared[j] = x[j];
    }
    
    // Barrier synchronization: wait until all threads finish writing to x_shared
    // before any warp begins computing its dot product
    __syncthreads();
    
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    // Each warp processes one unique row of the output
    int row = blockIdx.x * (blockDim.x / 32) + warpId;
    
    if (row < M) {
        float partial_sum = 0.0f;
        
        // Coalesced memory access for A, reading x from fast shared memory
        for (int j = laneId; j < N; j += 32) {
            partial_sum += A[(size_t)row * N + j] * x_shared[j];
        }
        
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
        
        // Lane 0 holds the final reduced sum for the row and writes to global memory
        if (laneId == 0) {
            y[row] = partial_sum;
        }
    }
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    
    // Calculate required shared memory size in bytes
    // For N up to 4096, this is at most 16KB, well within the typical 48KB block limit
    size_t shared_mem_bytes = N * sizeof(float);
    
    // Launch kernel with dynamic shared memory (the 3rd configuration parameter)
    gemv_shared_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}