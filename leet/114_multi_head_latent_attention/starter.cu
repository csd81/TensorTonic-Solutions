#include <cuda_runtime.h>
#include <math.h>

__global__ void mla_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ kv_cache,
    const float* __restrict__ W_UK,
    const float* __restrict__ W_UV,
    float* __restrict__ output,
    int num_heads,
    int seq_len,
    int kv_lora_rank,
    int head_dim,
    int rope_dim) 
{
    // Each block handles one head
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Derived stride variables
    int q_row_stride = head_dim + rope_dim;
    int kv_row_stride = kv_lora_rank + rope_dim;

    // Pointers specific to this head
    const float* q_head = q + h * q_row_stride;
    const float* q_nope = q_head;
    const float* q_pe   = q_head + head_dim;

    const float* W_UK_head = W_UK + h * head_dim * kv_lora_rank;
    const float* W_UV_head = W_UV + h * kv_lora_rank * head_dim;

    float scale = 1.0f / sqrtf((float)(head_dim + rope_dim));

    // Shared memory dynamically sized or structured dynamically for intermediate storage
    // Allocate max possible shared memory or rely on dynamically allocated shared memory
    extern __shared__ float sdata[];
    
    // We can partition sdata into:
    // 1. q_tilde: size kv_lora_rank
    // 2. scores/reductions buffer
    float* q_tilde = sdata; 

    // --- Step 1: Compute q_tilde = q_nope * W_UK_head ---
    // q_nope has shape (head_dim), W_UK_head has shape (head_dim, kv_lora_rank)
    // q_tilde has shape (kv_lora_rank)
    for (int col = tid; col < kv_lora_rank; col += bdim) {
        float sum = 0.0f;
        for (int row = 0; row < head_dim; ++row) {
            sum += q_nope[row] * W_UK_head[row * kv_lora_rank + col];
        }
        q_tilde[col] = sum;
    }
    __syncthreads();

    // --- Step 2: Compute Softmax Scores ---
    // Find Max Score for numerical stability
    float max_score = -1e20f;
    for (int t = tid; t < seq_len; t += bdim) {
        const float* kv_t = kv_cache + t * kv_row_stride;
        const float* c_t = kv_t;
        const float* k_pe_t = kv_t + kv_lora_rank;

        float score_nope = 0.0f;
        for (int i = 0; i < kv_lora_rank; ++i) {
            score_nope += q_tilde[i] * c_t[i];
        }

        float score_pe = 0.0f;
        for (int i = 0; i < rope_dim; ++i) {
            score_pe += q_pe[i] * k_pe_t[i];
        }

        float s_ht = (score_nope + score_pe) * scale;
        if (s_ht > max_score) {
            max_score = s_ht;
        }
    }

    // Block reduction for max_score
    for (int offset = 16; offset > 0; offset /= 2) {
        max_score = fmaxf(max_score, __shfl_xor_sync(0xffffffff, max_score, offset));
    }
    
    // Share max_score across warps using shared memory
    volatile __shared__ float shared_max;
    if (tid % 32 == 0) sdata[kv_lora_rank + (tid / 32)] = max_score;
    __syncthreads();
    if (tid < 32) {
        float m = (tid < (bdim / 32)) ? sdata[kv_lora_rank + tid] : -1e20f;
        for (int offset = 16; offset > 0; offset /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset));
        if (tid == 0) shared_max = m;
    }
    __syncthreads();
    max_score = shared_max;

    // Compute Exponentials Sum
    float exp_sum = 0.0f;
    for (int t = tid; t < seq_len; t += bdim) {
        const float* kv_t = kv_cache + t * kv_row_stride;
        const float* c_t = kv_t;
        const float* k_pe_t = kv_t + kv_lora_rank;

        float score_nope = 0.0f;
        for (int i = 0; i < kv_lora_rank; ++i) score_nope += q_tilde[i] * c_t[i];

        float score_pe = 0.0f;
        for (int i = 0; i < rope_dim; ++i) score_pe += q_pe[i] * k_pe_t[i];

        float s_ht = (score_nope + score_pe) * scale;
        exp_sum += expf(s_ht - max_score);
    }

    // Block reduction for exp_sum
    for (int offset = 16; offset > 0; offset /= 2) {
        exp_sum += __shfl_xor_sync(0xffffffff, exp_sum, offset);
    }
    if (tid % 32 == 0) sdata[kv_lora_rank + (tid / 32)] = exp_sum;
    __syncthreads();
    if (tid < 32) {
        float s = (tid < (bdim / 32)) ? sdata[kv_lora_rank + tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) s += __shfl_xor_sync(0xffffffff, s, offset);
        if (tid == 0) shared_max = s; // reuse shared_max space for shared_sum
    }
    __syncthreads();
    float sum_exps = shared_max;

    // --- Step 3: Compute Weighted Latent Sum (l) ---
    // l vector has length kv_lora_rank
    float* l_vec = sdata + kv_lora_rank + 32; // Offset safely past metadata buffers

    for (int col = tid; col < kv_lora_rank; col += bdim) {
        float l_element = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            const float* kv_t = kv_cache + t * kv_row_stride;
            const float* c_t = kv_t;
            const float* k_pe_t = kv_t + kv_lora_rank;

            float score_nope = 0.0f;
            for (int i = 0; i < kv_lora_rank; ++i) score_nope += q_tilde[i] * c_t[i];

            float score_pe = 0.0f;
            for (int i = 0; i < rope_dim; ++i) score_pe += q_pe[i] * k_pe_t[i];

            float a_ht = expf((score_nope + score_pe) * scale - max_score) / sum_exps;
            l_element += a_ht * c_t[col];
        }
        l_vec[col] = l_element;
    }
    __syncthreads();

    // --- Step 4: Apply Output Projection (out = l * W_UV) ---
    // l_vec has shape (kv_lora_rank), W_UV_head has shape (kv_lora_rank, head_dim)
    float* output_head = output + h * head_dim;
    for (int col = tid; col < head_dim; col += bdim) {
        float final_val = 0.0f;
        for (int row = 0; row < kv_lora_rank; ++row) {
            final_val += l_vec[row] * W_UV_head[row * head_dim + col];
        }
        output_head[col] = final_val;
    }
}

// Host entry function
extern "C" void solve(const float* q, const float* kv_cache, const float* W_UK, const float* W_UV,
                      float* output, int num_heads, int seq_len, int kv_lora_rank, int head_dim,
                      int rope_dim) 
{
    // Using 256 threads per block gives good execution balance for reduction tasks
    int threads_per_block = 256;
    int blocks_per_grid = num_heads;

    // Calculate dynamic shared memory footprint requirement
    // q_tilde (kv_lora_rank floats) + warp staging (32 floats) + l_vec (kv_lora_rank floats)
    size_t shared_mem_size = (kv_lora_rank + 32 + kv_lora_rank) * sizeof(float);

    mla_decode_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        q, kv_cache, W_UK, W_UV, output, num_heads, seq_len, kv_lora_rank, head_dim, rope_dim
    );
    
    cudaDeviceSynchronize();
}
