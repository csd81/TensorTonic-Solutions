#include <cuda_runtime.h>
#include <math.h>

// Optimized CUDA kernel for chunked-prefill over paged KV cache with GQA
__global__ void chunked_prefill_paged_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ context_lens,
    const int* __restrict__ cu_seqlens_q,
    float* __restrict__ output,
    int num_seqs,
    int total_q,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq) 
{
    // Global index maps to a unique query token and a specific query head
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = total_q * num_q_heads;

    if (global_id >= total_threads) return;

    // Decode current target query token index and query head index
    int q_idx = global_id / num_q_heads;
    int h = global_id % num_q_heads;

    // Map the packed query token index to its source sequence index
    int s = 0;
    // Sequential scan is optimal here for small to mid-sized batch constraints (num_seqs <= 64)
    for (int seq = 0; seq < num_seqs; ++seq) {
        if (q_idx >= cu_seqlens_q[seq] && q_idx < cu_seqlens_q[seq + 1]) {
            s = seq;
            break;
        }
    }

    // Determine spatial characteristics of the target sequence and token
    int L = context_lens[s];
    int q_start = cu_seqlens_q[s];
    int i = q_idx - q_start; // Local position relative to chunk start
    int p_i = L - (cu_seqlens_q[s + 1] - q_start) + i; // Global causal window ceiling

    // Compute matching KV head using integer division mapping (GQA)
    int g = h / (num_q_heads / num_kv_heads);
    float scale = 1.0f / sqrtf((float)head_dim);

    // Compute pointer offset to the specific query vector head
    const float* q_ptr = Q + (size_t)q_idx * num_q_heads * head_dim + h * head_dim;

    // --- Pass 1: Compute raw attention scores and find max_score for numerical stability ---
    float max_score = -1e20f;
    
    // We can use a small stack allocation or temporary loops if context memory constraints exist.
    // Given arbitrary lengths up to 8,192, we balance performance by looping or managing global scores.
    // To ensure float32 fidelity and zero library reliance, we run a unified pass over the logical sequence tokens up to p_i.
    for (int t = 0; t <= p_i; ++t) {
        int block_idx = block_table[s * max_blocks_per_seq + (t / block_size)];
        int block_offset = t % block_size;
        
        // Target physical offset within the scattered shared paged pool
        size_t k_offset = ((size_t)block_idx * block_size * num_kv_heads * head_dim) + 
                          ((size_t)block_offset * num_kv_heads * head_dim) + 
                          ((size_t)g * head_dim);
        const float* k_ptr = K_cache + k_offset;

        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        if (score > max_score) {
            max_score = score;
        }
    }

    // --- Pass 2: Calculate denominator sum of exponentials (Softmax Normalizer) ---
    float exp_sum = 0.0f;
    for (int t = 0; t <= p_i; ++t) {
        int block_idx = block_table[s * max_blocks_per_seq + (t / block_size)];
        int block_offset = t % block_size;

        size_t k_offset = ((size_t)block_idx * block_size * num_kv_heads * head_dim) + 
                          ((size_t)block_offset * num_kv_heads * head_dim) + 
                          ((size_t)g * head_dim);
        const float* k_ptr = K_cache + k_offset;

        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        exp_sum += expf(score - max_score);
    }

    // --- Pass 3: Multiply normalized soft weights against Values and write back output ---
    float* out_ptr = output + (size_t)q_idx * num_q_heads * head_dim + h * head_dim;
    
    // Clear / Initialize the specific output tensor dimensions allocated for this thread
    for (int d = 0; d < head_dim; ++d) {
        out_ptr[d] = 0.0f;
    }

    for (int t = 0; t <= p_i; ++t) {
        int block_idx = block_table[s * max_blocks_per_seq + (t / block_size)];
        int block_offset = t % block_size;

        // Fetch score again for calculating soft attention probability
        size_t k_offset = ((size_t)block_idx * block_size * num_kv_heads * head_dim) + 
                          ((size_t)block_offset * num_kv_heads * head_dim) + 
                          ((size_t)g * head_dim);
        const float* k_ptr = K_cache + k_offset;

        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        float attn_weight = expf(score - max_score) / exp_sum;

        // Fetch corresponding Value vector from paged cache
        size_t v_offset = ((size_t)block_idx * block_size * num_kv_heads * head_dim) + 
                          ((size_t)block_offset * num_kv_heads * head_dim) + 
                          ((size_t)g * head_dim);
        const float* v_ptr = V_cache + v_offset;

        // Vector accumulation phase
        for (int d = 0; d < head_dim; ++d) {
            out_ptr[d] += attn_weight * v_ptr[d];
        }
    }
}

// C-interface host wrapper function
extern "C" void solve(const float* Q, const float* K_cache, const float* V_cache,
                      const int* block_table, const int* context_lens, const int* cu_seqlens_q,
                      float* output, int num_seqs, int total_q, int num_q_heads, int num_kv_heads,
                      int head_dim, int block_size, int max_blocks_per_seq) 
{
    int total_threads = total_q * num_q_heads;
    if (total_threads == 0) return;

    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    chunked_prefill_paged_attention_kernel<<<blocks_per_grid, threads_per_block>>>(
        Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, output,
        num_seqs, total_q, num_q_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq
    );

    // Sync line to catch execution issues if tracking via host-side runtime wrappers
    cudaDeviceSynchronize();
}
