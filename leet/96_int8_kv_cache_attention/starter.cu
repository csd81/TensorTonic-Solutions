#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

// Shared memory size can be adjusted or dynamically allocated if needed, 
// but for stable warp reductions, standard register shuffling is preferred.
__global__ void decode_kv_int8_attention_kernel(
    const float* __restrict__ Q,
    const int8_t* __restrict__ K_int8,
    const int8_t* __restrict__ V_int8,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    float* __restrict__ output,
    int num_heads,
    int seq_len,
    int head_dim) 
{
    // Each block handles one head
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Scale factor for scaled dot-product attention
    float scale_factor = 1.0f / sqrtf((float)head_dim);

    // Online softmax tracking variables
    float max_score = -CUDART_INF_F;
    float sum_exp = 0.0f;

    // Dynamically allocated or statically bounded accumulator for output
    // Maximum head_dim constraint is 256. Each thread accumulates parts of the output.
    // To handle arbitrary head_dim up to 256 dynamically per thread:
    float out_acc[256] = {0.0f};

    // 1. Compute dot products and track online softmax
    // Iterate over the sequence dimension
    for (int s = tid; s < seq_len; s += bdim) {
        float score = 0.0f;
        
        // Compute Q * K^T for token `s` and head `h`
        for (int d = 0; d < head_dim; ++d) {
            float q_val = Q[h * head_dim + d];
            // K_int8 layout: [num_heads, seq_len, head_dim]
            int8_t k_i8 = K_int8[(h * seq_len + s) * head_dim + d];
            // k_scale layout: [num_heads, seq_len]
            float k_s = k_scale[h * seq_len + s];
            
            float k_val = (float)k_i8 * k_s;
            score += q_val * k_val;
        }
        score *= scale_factor;

        // Online Softmax updates
        float old_max = max_score;
        if (score > max_score) {
            max_score = score;
            float exp_diff = expf(old_max - max_score);
            sum_exp *= exp_diff;
            for (int d = 0; d < head_dim; ++d) {
                out_acc[d] *= exp_diff;
            }
        }

        float weight = expf(score - max_score);
        sum_exp += weight;

        // Accumulate V_float * weight
        float v_s = v_scale[h * seq_len + s];
        for (int d = 0; d < head_dim; ++d) {
            int8_t v_i8 = V_int8[(h * seq_len + s) * head_dim + d];
            float v_val = (float)v_i8 * v_s;
            out_acc[d] += weight * v_val;
        }
    }

    // 2. Inter-thread reduction within the block using shared memory
    // Allocate shared memory for block-level reduction
    extern __shared__ float shared_mem[];
    float* smem_max = &shared_mem[0];         // size: bdim
    float* smem_sum = &shared_mem[bdim];      // size: bdim
    float* smem_out = &shared_mem[2 * bdim]; // size: bdim * head_dim

    smem_max[tid] = max_score;
    smem_sum[tid] = sum_exp;
    for (int d = 0; d < head_dim; ++d) {
        smem_out[tid * head_dim + d] = out_acc[d];
    }
    __syncthreads();

    // Tree reduction over the block threads
    for (int stride = bdim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            int remote = tid + stride;
            float max_left = smem_max[tid];
            float max_right = smem_max[remote];
            
            float new_max = (max_left > max_right) ? max_left : max_right;
            float exp_left = expf(max_left - new_max);
            float exp_right = expf(max_right - new_max);

            float new_sum = smem_sum[tid] * exp_left + smem_sum[remote] * exp_right;

            for (int d = 0; d < head_dim; ++d) {
                float out_left = smem_out[tid * head_dim + d] * exp_left;
                float out_right = smem_out[remote * head_dim + d] * exp_right;
                smem_out[tid * head_dim + d] = out_left + out_right;
            }

            smem_max[tid] = new_max;
            smem_sum[tid] = new_sum;
        }
        __syncthreads();
    }

    // 3. Thread 0 writes final normalized results to global output
    if (tid == 0) {
        float final_sum = smem_sum[0];
        float inv_sum = (final_sum != 0.0f) ? (1.0f / final_sum) : 0.0f;

        for (int d = 0; d < head_dim; ++d) {
            output[h * head_dim + d] = smem_out[d] * inv_sum;
        }
    }
}

extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) 
{
    // Define block and grid configurations
    int threads_per_block = 256; 
    int blocks_per_grid = num_heads;

    // Calculate dynamic shared memory requirements:
    // smem_max (threads) + smem_sum (threads) + smem_out (threads * head_dim)
    size_t shared_mem_size = (2 * threads_per_block * sizeof(float)) + 
                             (threads_per_block * head_dim * sizeof(float));

    decode_kv_int8_attention_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim
    );
}
