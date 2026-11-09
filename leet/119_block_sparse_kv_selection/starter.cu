#include <cuda_runtime.h>
#include <math.h>

// Helper macro for safe indexing or ceiling division
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void sparse_attention_decode_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ output, 
    int num_heads, int seq_len, int head_dim, int block_size, int num_selected) 
{
    // Each CUDA block processes one attention head
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int num_blocks = CEIL_DIV(seq_len, block_size);
    int k_selected = (num_selected < num_blocks) ? num_selected : num_blocks;

    // Dynamically allocated shared memory inside kernel
    // Max constraints: num_blocks <= 8192 / 1 = 8192 if block_size=1.
    // To remain safely bounded within typical shared memory limits, we can 
    // allocate arrays dynamically or use a fixed size up to the max possible blocks.
    // However, since 8192 floats is 32KB, we can use an external or large enough shared array.
    extern __shared__ float shared_mem[];
    
    // Layout: 
    // scores: [num_blocks]
    // block_indices: [num_blocks]
    float* scores = shared_mem;
    int* block_indices = (int*)(scores + num_blocks);

    // Step 1: Compute mean-pooled keys and scores for each block
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        int start_t = b * block_size;
        int end_t = (start_t + block_size < seq_len) ? (start_t + block_size) : seq_len;
        int actual_block_size = end_t - start_t;

        // Compute dot product of Q[h] and mean(K[h, start_t:end_t])
        float dot_sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float q_val = Q[h * head_dim + d];
            float k_sum = 0.0f;
            for (int t = start_t; t < end_t; ++t) {
                k_sum += K[(h * seq_len + t) * head_dim + d];
            }
            dot_sum += q_val * (k_sum / actual_block_size);
        }
        scores[b] = dot_sum;
        block_indices[b] = b;
    }

    __syncthreads();

    // Step 2: Select the top min(num_selected, num_blocks) blocks deterministically
    // Using a single thread or simple selection sort on shared memory since num_blocks is small
    if (threadIdx.x == 0) {
        for (int i = 0; i < k_selected; ++i) {
            int best_idx = i;
            for (int j = i + 1; j < num_blocks; ++j) {
                // If score is higher, or scores are equal but index is smaller
                if (scores[j] > scores[best_idx] || 
                   (scores[j] == scores[best_idx] && block_indices[j] < block_indices[best_idx])) {
                    best_idx = j;
                }
            }
            // Swap scores
            float tmp_score = scores[i];
            scores[i] = scores[best_idx];
            scores[best_idx] = tmp_score;
            // Swap indices
            int tmp_idx = block_indices[i];
            block_indices[i] = block_indices[best_idx];
            block_indices[best_idx] = tmp_idx;
        }
    }

    __syncthreads();

    // Step 3: Compute attention map over the positions of the selected blocks
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Find max score for numerical stability in softmax
    // Reuse shared memory to store unnormalized attention weights per thread
    // Max possible positions selected = k_selected * block_size
    int total_selected_positions = 0;
    for (int i = 0; i < k_selected; ++i) {
        int b = block_indices[i];
        int start_t = b * block_size;
        int end_t = (start_t + block_size < seq_len) ? (start_t + block_size) : seq_len;
        total_selected_positions += (end_t - start_t);
    }

    // Parallel Max Finder over selected positions
    float max_attn_score = -1e20f;
    for (int idx = threadIdx.x; idx < total_selected_positions; idx += blockDim.x) {
        // Map global index back to the specific selected position
        int temp_idx = idx;
        int target_t = -1;
        for (int i = 0; i < k_selected; ++i) {
            int b = block_indices[i];
            int start_t = b * block_size;
            int end_t = (start_t + block_size < seq_len) ? (start_t + block_size) : seq_len;
            int len = end_t - start_t;
            if (temp_idx < len) {
                target_t = start_t + temp_idx;
                break;
            }
            temp_idx -= len;
        }

        // Q * K dot product for this position
        float qk = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            qk += Q[h * head_dim + d] * K[(h * seq_len + target_t) * head_dim + d];
        }
        qk *= scale;
        if (qk > max_attn_score) max_attn_score = qk;
    }

    // Warp / Block Reduction for max_attn_score
    // Using simple shared memory layout to reduce or shared registers
    extern __shared__ float shared_reduce[]; 
    // To avoid collision with step 1 allocations, we can offset it
    float* block_max = (float*)(shared_mem + 2 * num_blocks);
    block_max[threadIdx.x] = max_attn_score;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (block_max[threadIdx.x + stride] > block_max[threadIdx.x]) {
                block_max[threadIdx.x] = block_max[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    max_attn_score = block_max[0];

    // Compute denominator (Sum of Exp)
    float exp_sum = 0.0f;
    for (int idx = threadIdx.x; idx < total_selected_positions; idx += blockDim.x) {
        int temp_idx = idx;
        int target_t = -1;
        for (int i = 0; i < k_selected; ++i) {
            int b = block_indices[i];
            int start_t = b * block_size;
            int end_t = (start_t + block_size < seq_len) ? (start_t + block_size) : seq_len;
            int len = end_t - start_t;
            if (temp_idx < len) {
                target_t = start_t + temp_idx;
                break;
            }
            temp_idx -= len;
        }

        float qk = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            qk += Q[h * head_dim + d] * K[(h * seq_len + target_t) * head_dim + d];
        }
        qk *= scale;
        exp_sum += expf(qk - max_attn_score);
    }

    float* block_sum = (float*)(shared_mem + 2 * num_blocks);
    block_sum[threadIdx.x] = exp_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    exp_sum = block_sum[0];

    // Step 4: Weighted sum over V tensor to compute final output
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        int temp_idx = 0;
        
        for (int i = 0; i < k_selected; ++i) {
            int b = block_indices[i];
            int start_t = b * block_size;
            int end_t = (start_t + block_size < seq_len) ? (start_t + block_size) : seq_len;
            
            for (int t = start_t; t < end_t; ++t) {
                float qk = 0.0f;
                for (int cd = 0; cd < head_dim; ++cd) {
                    qk += Q[h * head_dim + cd] * K[(h * seq_len + t) * head_dim + cd];
                }
                qk *= scale;
                float weight = expf(qk - max_attn_score) / exp_sum;
                out_val += weight * V[(h * seq_len + t) * head_dim + d];
            }
        }
        output[h * head_dim + d] = out_val;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int num_heads,
                      int seq_len, int head_dim, int block_size, int num_selected) 
{
    int num_blocks = CEIL_DIV(seq_len, block_size);
    
    // Allocate dynamic shared memory required for scores, indices and reduction buffers
    size_t shared_mem_size = (2 * num_blocks * sizeof(float)) + (num_blocks * sizeof(int)) + (256 * sizeof(float));

    // 256 threads per block gives a robust execution balance for dimensionality transitions
    int threads_per_block = 256;
    int blocks_per_grid = num_heads;

    sparse_attention_decode_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        Q, K, V, output, num_heads, seq_len, head_dim, block_size, num_selected
    );
}
