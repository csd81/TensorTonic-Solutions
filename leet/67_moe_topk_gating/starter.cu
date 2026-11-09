#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Shared memory size allocated dynamically or statically bound to max E (256)
__global__ void topk_gating_kernel(const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    
    // Shared memory to store logits and corresponding initial indices
    __shared__ float s_logits[256];
    __shared__ int s_indices[256];

    // 1. Cooperative load into Shared Memory
    if (tid < E) {
        s_logits[tid] = logits[row * E + tid];
        s_indices[tid] = tid;
    } else {
        s_logits[tid] = -FLT_MAX; // Padding for non-existent elements up to block size
        s_indices[tid] = -1;
    }
    __syncthreads();

    // 2. Selection Sort via Threads (Since E <= 256 and k is typically small like 2)
    // Thread 0 will perform an inline selection to find top-k components sequentially.
    // This avoids sorting overhead when k is small.
    if (tid == 0) {
        for (int i = 0; i < k; ++i) {
            int max_idx = i;
            float max_val = s_logits[i];
            for (int j = i + 1; j < E; ++j) {
                if (s_logits[j] > max_val) {
                    max_val = s_logits[j];
                    max_idx = j;
                }
            }
            // Swap values
            if (max_idx != i) {
                float tmp_f = s_logits[i];
                s_logits[i] = s_logits[max_idx];
                s_logits[max_idx] = tmp_f;

                int tmp_i = s_indices[i];
                s_indices[i] = s_indices[max_idx];
                s_indices[max_idx] = tmp_i;
            }
        }
    }
    __syncthreads();

    // 3. Thread-parallel Softmax over the extracted Top-K elements
    // Find Max (for stable Softmax numerical defense)
    __shared__ float max_logit;
    if (tid == 0) {
        max_logit = s_logits[0]; 
    }
    __syncthreads();

    // Compute Exponentials & Sum
    __shared__ float s_exp[256];
    if (tid < k) {
        s_exp[tid] = expf(s_logits[tid] - max_logit);
    } else {
        s_exp[tid] = 0.0f;
    }
    __syncthreads();

    // Parallel Reduction inside block to sum exponents
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_exp[tid] += s_exp[tid + stride];
        }
        __syncthreads();
    }

    // Write back final result values to global arrays
    if (tid < k) {
        float sum_exp = s_exp[0];
        topk_indices[row * k + tid] = s_indices[tid];
        topk_weights[row * k + tid] = expf(s_logits[tid] - max_logit) / sum_exp;
    }
}

extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k) {
    // 1 block per row, 256 threads per block since E <= 256
    int threadsPerBlock = 256;
    int blocksPerGrid = M;

    topk_gating_kernel<<<blocksPerGrid, threadsPerBlock>>>(logits, topk_weights, topk_indices, M, E, k);
    
    // Ensure accurate synchronization if required externally
    cudaDeviceSynchronize();
}
