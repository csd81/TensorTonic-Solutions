#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// CUDA kernel to compute partial dot products with FP32 precision
__global__ void dotProductKernel(const half* A, const half* B, float* block_outputs, int N) {
    // Shared memory for block-level reduction
    __shared__ float cache[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Grid-stride loop to handle large N efficiently
    while (i < N) {
        // Convert __half to float before multiplication for FP32 accumulation precision
        float a_val = __half2float(A[i]);
        float b_val = __half2float(B[i]);
        sum += a_val * b_val;
        i += blockDim.x * gridDim.x;
    }
    
    // Copy local sum to shared memory
    cache[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    
    // Write the block's reduction result to global memory
    if (tid == 0) {
        block_outputs[blockIdx.x] = cache[0];
    }
}

// Final reduction kernel to sum block outputs and convert the result back to half
__global__ void finalReductionKernel(const float* block_outputs, half* result, int num_blocks) {
    float total_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_sum += block_outputs[i];
    }
    // Convert final FP32 accumulated result back to half
    *result = __float2half(total_sum);
}

// Host function
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    // Determine grid size (capped at a reasonable limit to prevent excessive allocation)
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 2048) {
        num_blocks = 2048; // Grid stride loop handles the remaining elements
    }
    
    // Allocate temporary device memory for block outputs
    float* d_block_outputs;
    cudaMalloc(&d_block_outputs, num_blocks * sizeof(float));
    
    // Launch primary dot product kernel
    dotProductKernel<<<num_blocks, threads_per_block>>>(A, B, d_block_outputs, N);
    
    // Launch final reduction kernel to sum all block totals into the final half result
    finalReductionKernel<<<1, 1>>>(d_block_outputs, result, num_blocks);
    
    // Clean up temporary memory
    cudaFree(d_block_outputs);
}
