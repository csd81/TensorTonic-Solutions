#include <cuda_runtime.h>

// CUDA Kernel to compute partial sums using block-level reduction
__global__ void subarray_sum_kernel(const int* input, int* output, int S, int E) {
    // Each block accumulates its sum in shared memory
    extern __shared__ int shared_data[];
    
    int block_sum = 0;
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    // Grid-stride loop over the requested subarray range [S..E]
    for (int i = S + global_id; i <= E; i += grid_size) {
        block_sum += input[i];
    }

    // Store thread sum into shared memory
    shared_data[tid] = block_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Accumulate the block's total sum into the global output variable
    if (tid == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    // 1. Initialize output to 0 on the device
    cudaMemset(output, 0, sizeof(int));

    // Calculate total elements to process
    int elements_to_process = E - S + 1;
    if (elements_to_process <= 0) {
        return;
    }

    // 2. Configure Execution Parameters
    int threads_per_block = 256;
    // Cap the number of blocks to avoid launching unnecessary empty waves
    int blocks_per_grid = (elements_to_process + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 2048) {
        blocks_per_grid = 2048; // Sufficient to saturate GPU warps
    }

    // Amount of shared memory allocated dynamically per block (in bytes)
    size_t shared_mem_size = threads_per_block * sizeof(int);

    // 3. Launch the Kernel
    subarray_sum_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        input, output, S, E
    );

    // Synchronize to guarantee completion before returning
    cudaDeviceSynchronize();
}
