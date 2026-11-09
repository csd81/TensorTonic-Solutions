#include <cuda_runtime.h>
#include <math.h>

// CUDA Kernel executing shared-prefix causal attention
__global__ void cascade_attention_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K_prefix, 
    const float* __restrict__ V_prefix,
    const float* __restrict__ K_suffix, 
    const float* __restrict__ V_suffix, 
    float* __restrict__ output, 
    int batch_size, int num_heads, int prefix_len, int suffix_len, int head_dim) 
{
    // Each block/thread combination maps to a single sequence sequence query item
    // For large sequence lengths, threads handle single suffix query lines
    int b = blockIdx.y;
    int h = blockIdx.x;
    int i = threadIdx.x + blockIdx.z * blockDim.x;

    if (i >= suffix_len) return;

    // Pre-calculate scale factor 1 / sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);

    // Compute pointer base offsets
    int head_stride_suffix = suffix_len * head_dim;
    int batch_stride_suffix = num_heads * head_stride_suffix;
    
    int prefix_head_stride = prefix_len * head_dim;

    // Locating the exact rows
    const float* q_ptr = Q + (b * batch_stride_suffix) + (h * head_stride_suffix) + (i * head_dim);
    const float* k_pref_base = K_prefix + (h * prefix_head_stride);
    const float* v_pref_base = V_prefix + (h * prefix_head_stride);
    const float* k_suff_base = K_suffix + (b * batch_stride_suffix) + (h * head_stride_suffix);
    const float* v_suff_base = V_suffix + (b * batch_stride_suffix) + (h * head_stride_suffix);
    float* out_ptr = output + (b * batch_stride_suffix) + (h * head_stride_suffix) + (i * head_dim);

    // FlashAttention-style Online Softmax running states
    float m = -CUDART_INF_F;
    float d = 0.0f;

    // Dynamically tracking accumulated values using register array up to maximum dimension size
    // head_dim max constraints: 128
    float acc[128];
    for (int d_idx = 0; d_idx < head_dim; ++d_idx) {
        acc[d_idx] = 0.0f;
    }

    // Phase 1: Process Shared Prefix Context
    for (int j = 0; j < prefix_len; ++j) {
        float score = 0.0f;
        const float* k_ptr = k_pref_base + (j * head_dim);
        for (int d_idx = 0; d_idx < head_dim; ++d_idx) {
            score += q_ptr[d_idx] * k_ptr[d_idx];
        }
        score *= scale;

        // Maintain Online Softmax Numerators & Denominators
        if (score > m) {
            float e = expf(m - score);
            for (int d_idx = 0; d_idx < head_dim; ++d_idx) {
                acc[d_idx] *= e;
            }
            d = d * e + 1.0f;
            m = score;
        } else {
            float e = expf(score - m);
            d += e;
            const float* v_ptr = v_pref_base + (j * head_dim);
            for (int d_idx = 0; d_idx < head_dim; ++d_idx) {
                acc[d_idx] += e * v_ptr[d_idx];
            }
            continue;
        }

        const float* v_ptr = v_pref_base + (j * head_dim);
        for (int d_idx = 0; d_idx < head_dim; ++d_idx) {
            acc[d_idx] += v_ptr[d_idx];
        }
    }

    // Phase 2: Process Local Sequence Causal Suffix (0 inclusive to i)
    for (int j = 0; j <= i; ++j) {
        float score = 0.0f;
        const float* k_ptr = k_suff_base + (j * head_dim);
        for (int d_idx = 0; d_idx < head_dim; ++d_idx) {
            score += q_ptr[d_idx] * k_ptr[d_idx];
        }
        score *= scale;

        // Maintain Online Softmax Numerators & Denominators
        if (score > m) {
            float e = expf(m - score);
            for (int d_idx = 0; d_idx < head_dim; ++d_idx) {
                acc[d_idx] *= e;
            }
            d = d * e + 1.0f;
            m = score;
        } else {
            float e = expf(score - m);
            d += e;
            const float* v_ptr = v_suff_base + (j * head_dim);
            for (int d_idx = 0; d_idx < head_dim; ++d_idx) {
                acc[d_idx] += e * v_ptr[d_idx];
            }
            continue;
        }

        const float* v_ptr = v_suff_base + (j * head_dim);
        for (int d_idx = 0; d_idx < head_dim; ++d_idx) {
            acc[d_idx] += v_ptr[d_idx];
        }
    }

    // Step 3: Normalize with denominator pool and save results to Global Memory
    float inv_d = 1.0f / d;
    for (int d_idx = 0; d_idx < head_dim; ++d_idx) {
        out_ptr[d_idx] = acc[d_idx] * inv_d;
    }
}

extern "C" void solve(const float* Q, const float* K_prefix, const float* V_prefix,
                      const float* K_suffix, const float* V_suffix, float* output, int batch_size,
                      int num_heads, int prefix_len, int suffix_len, int head_dim) 
{
    // Define structural thread mapping
    int threads_per_block = 256;
    int blocks_z = (suffix_len + threads_per_block - 1) / threads_per_block;

    // Grid Mapping: X -> heads, Y -> batch sequences, Z -> chunk segments of suffix lines
    dim3 grid(num_heads, batch_size, blocks_z);
    dim3 block(threads_per_block);

    cascade_attention_kernel<<<grid, block>>>(
        Q, K_prefix, V_prefix, K_suffix, V_suffix, output,
        batch_size, num_heads, prefix_len, suffix_len, head_dim
    );
}
