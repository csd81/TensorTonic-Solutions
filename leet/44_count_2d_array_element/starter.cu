#include <cuda_runtime.h>

// Kernel that counts occurrences of K using shared memory reduction
__global__ void count_kernel(const int* input, int* output, int total_elements, int K) {
    // Shared memory for block-level reduction (blockDim.x threads)
    extern __shared__ int shared_counts[];

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_count = 0;

    // Grid-stride loop to handle any array size dynamically
    for (int i = global_id; i < total_elements; i += stride) {
        if (input[i] == K) {
            local_count++;
        }
    }

    // Store the local thread count into shared memory
    shared_counts[tid] = local_count;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_counts[tid] += shared_counts[tid + s];
        }
        __syncthreads();
    }

    // The first thread in the block adds the block's total to the global output
    if (tid == 0) {
        atomicAdd(output, shared_counts[0]);
    }
}

extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    long long total_elements = (long long)N * M;
    
    // Initialize the output memory on the device to 0
    cudaMemset(output, 0, sizeof(int));

    // Define execution configuration
    int threads_per_block = 256;
    
    // Calculate grid size, capping it to avoid launching unnecessary blocks
    long long required_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    int blocks_per_grid = (required_blocks > 4096) ? 4096 : required_blocks;

    // Dynamically allocate shared memory size based on the number of threads per block
    size_t shared_mem_size = threads_per_block * sizeof(int);

    // Launch the kernel
    count_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        input, 
        output, 
        total_elements, 
        K
    );

    // Wait for the GPU to finish execution
    cudaDeviceSynchronize();
}
