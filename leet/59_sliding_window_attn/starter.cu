#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel to compute sliding window attention for each query row
__global__ void sliding_window_attn_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ output, 
    int M, 
    int d, 
    int window_size) 
{
    // Each block handles one row (query index i)
    int i = blockIdx.x;
    if (i >= M) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Determine window boundaries for query position i
    int start_j = max(0, i - window_size);
    int end_j = min(M - 1, i + window_size);
    int local_window_len = end_j - start_j + 1;

    // Use shared memory to store intermediate results for reductions
    // Max capacity needed: 2 * window_size + 1 (at most 65 floats)
    extern __shared__ float s_mem[];
    float* s_scores = s_mem;

    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    // Step 1: Compute dot products and find the maximum score for softmax stability
    float local_max = -1e20f;

    for (int idx = tid; idx < local_window_len; idx += bdim) {
        int j = start_j + idx;
        float score = 0.0f;
        
        // Compute dot product Q_i * K_j
        for (int k = 0; k < d; ++k) {
            score += Q[i * d + k] * K[j * d + k];
        }
        score *= inv_sqrt_d;
        
        s_scores[idx] = score;
        if (score > local_max) {
            local_max = score;
        }
    }

    // Shared memory reduction to find the global maximum within the block
    // Using simple block scan/reduction via shared memory
    __shared__ float s_block_max;
    if (tid == 0) s_block_max = -1e20f;
    __syncthreads();

    // AtomicMax or simple sequential reduction since block sizes are typically small (e.g., 128)
    // Here we use atomicMax on a shared float via an integer casting wrapper
    if (local_max != -1e20f) {
        int* address_as_i = (int*)&s_block_max;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            if (__int_as_float(assumed) >= local_max) break;
            old = atomicCAS(address_as_i, assumed, __float_as_int(local_max));
        } while (assumed != old);
    }
    __syncthreads();
    local_max = s_block_max;

    // Step 2: Compute exponential sums for the denominator of the softmax
    float local_sum = 0.0f;
    for (int idx = tid; idx < local_window_len; idx += bdim) {
        float exp_score = expf(s_scores[idx] - local_max);
        s_scores[idx] = exp_score; // Store back the unnormalized exp values
        local_sum += exp_score;
    }

    __shared__ float s_block_sum;
    if (tid == 0) s_block_sum = 0.0f;
    __syncthreads();

    if (local_sum > 0.0f) {
        atomicAdd(&s_block_sum, local_sum);
    }
    __syncthreads();
    float total_sum = s_block_sum;

    // Step 3: Compute the final weighted sum of values for each feature dimension
    for (int k = tid; k < d; k += bdim) {
        float out_val = 0.0f;
        for (int idx = 0; idx < local_window_len; ++idx) {
            int j = start_j + idx;
            float weight = s_scores[idx] / total_sum;
            out_val += weight * V[j * d + k];
        }
        output[i * d + k] = out_val;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int window_size) 
{
    // Select an optimal block size (e.g., 128 threads per block)
    int threadsPerBlock = 128;
    int blocksPerGrid = M;

    // Dynamic shared memory sizing: requires space for (2 * window_size + 1) floats
    size_t sharedMemSize = (2 * window_size + 1) * sizeof(float);

    sliding_window_attn_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        Q, K, V, output, M, d, window_size
    );
    
    // Sync to guarantee execution bounds complete cleanly
    cudaDeviceSynchronize();
}
```

<FollowUp>
If you want to optimize this kernel further, let me know:
* Do you want to try a **Warp Shuffle reduction** strategy to fully eliminate atomic operations?
* Would you like to evaluate implementing this via **Tensor Cores** if your target architecture supports it?
</FollowUp>
