#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel to merge per-chunk Flash-Decoding partial results
__global__ void flash_decoding_merge_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_lse,
    float* __restrict__ output,
    int num_splits,
    int num_heads,
    int head_dim)
{
    // Map each block to a single head
    int h = blockIdx.x;
    if (h >= num_heads) return;

    // Map each thread to an embedding dimension index
    int d = threadIdx.x;
    if (d >= head_dim) return;

    // 1. Find the maximum log-sum-exp value (m_h) across all splits for this head
    float m_h = -INFINITY;
    for (int s = 0; s < num_splits; ++s) {
        // partial_lse shape: (num_splits, num_heads)
        float lse = partial_lse[s * num_heads + h];
        if (lse > m_h) {
            m_h = lse;
        }
    }

    // 2. Accumulate the weighted partial outputs and compute total scale weight
    float sum_weights = 0.0f;
    float accumulated_out = 0.0f;

    for (int s = 0; s < num_splits; ++s) {
        float lse = partial_lse[s * num_heads + h];
        float w = __expf(lse - m_h); // Numerically stable exponentiation
        sum_weights += w;

        // partial_out shape: (num_splits, num_heads, head_dim)
        int out_idx = (s * num_heads + h) * head_dim + d;
        accumulated_out += w * partial_out[out_idx];
    }

    // 3. Compute final per-split-weighted average and store result
    // output shape: (num_heads, head_dim)
    if (sum_weights > 0.0f) {
        output[h * head_dim + d] = accumulated_out / sum_weights;
    } else {
        output[h * head_dim + d] = 0.0f;
    }
}

extern "C" void solve(const float* partial_out, const float* partial_lse, float* output,
                      int num_splits, int num_heads, int head_dim) 
{
    // Set execution grid configuration
    // 1 block per head, 1 thread per embedding dimension dimension
    dim3 threadsPerBlock(head_dim);
    dim3 numBlocks(num_heads);

    // Dynamic handling if head_dim exceeds max threads per block configuration (e.g., > 1024)
    if (head_dim > 1024) {
        threadsPerBlock.x = 256;
        // Adjust grid structure or internal thread loops if mapping arbitrary bounds
    }

    flash_decoding_merge_kernel<<<numBlocks, threadsPerBlock>>>(
        partial_out, partial_lse, output, num_splits, num_heads, head_dim
    );
}
