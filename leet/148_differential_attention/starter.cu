#include <cuda_runtime.h>
#include <math.h>

// Each block processes a single (query_idx, head_idx) pair.
// Parallelized across the dimension of V/output (2 * head_dim) using threads.
__global__ void diff_attn_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    const float* __restrict__ lambdas,
    float* __restrict__ output, 
    int seq_len, 
    int num_heads, 
    int head_dim, 
    float lambda_init) 
{
    int i = blockIdx.x; // Query sequence index (0 to seq_len - 1)
    int h = blockIdx.y; // Head index (0 to num_heads - 1)
    int tid = threadIdx.x;
    int v_dim = 2 * head_dim;

    // Strided loop setup over output features to remain adaptable to any block size
    int num_threads = blockDim.x;

    float scale = 1.0f / sqrtf((float)head_dim);
    float lambda_h = lambdas[h];

    // --- Step 1: Find max score for numerical stability in Softmax ---
    float max_score1 = -1e30f;
    float max_score2 = -1e30f;

    // Temporal pointers for this specific head
    const float* q_ptr = Q + (i * num_heads + h) * v_dim;
    
    // We iterate over all valid causal keys (t <= i)
    for (int t = 0; t <= i; ++t) {
        const float* k_ptr = K + (t * num_heads + h) * v_dim;
        
        // Compute dot products for both halves
        float score1 = 0.0f;
        float score2 = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score1 += q_ptr[d] * k_ptr[d];
            score2 += q_ptr[head_dim + d] * k_ptr[head_dim + d];
        }
        score1 *= scale;
        score2 *= scale;

        if (score1 > max_score1) max_score1 = score1;
        if (score2 > max_score2) max_score2 = score2;
    }

    // --- Step 2: Compute Softmax Denominators (Sum of Exponentials) ---
    float sum_exp1 = 0.0f;
    float sum_exp2 = 0.0f;
    for (int t = 0; t <= i; ++t) {
        const float* k_ptr = K + (t * num_heads + h) * v_dim;
        
        float score1 = 0.0f;
        float score2 = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score1 += q_ptr[d] * k_ptr[d];
            score2 += q_ptr[head_dim + d] * k_ptr[head_dim + d];
        }
        score1 *= scale;
        score2 *= scale;

        sum_exp1 += expf(score1 - max_score1);
        sum_exp2 += expf(score2 - max_score2);
    }

    // --- Step 3: Accumulate attention output over V features parallelly ---
    // Every thread manages a subset of the 2 * head_dim features of O[i]
    for (int feat = tid; feat < v_dim; feat += num_threads) {
        float o_feat = 0.0f;

        for (int t = 0; t <= i; ++t) {
            const float* k_ptr = K + (t * num_heads + h) * v_dim;
            const float* v_ptr = V + (t * num_heads + h) * v_dim;

            float score1 = 0.0f;
            float score2 = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score1 += q_ptr[d] * k_ptr[d];
                score2 += q_ptr[head_dim + d] * k_ptr[head_dim + d];
            }
            score1 *= scale;
            score2 *= scale;

            float a1 = expf(score1 - max_score1) / sum_exp1;
            float a2 = expf(score2 - max_score2) / sum_exp2;

            o_feat += (a1 - lambda_h * a2) * v_ptr[feat];
        }

        // Temporarily park O[i][feat] directly into the output tensor space 
        // to evaluate variance across features for RMSNorm
        output[(i * num_heads + h) * v_dim + feat] = o_feat;
    }

    // Synchronize to make sure all elements of O[i] are written to global memory
    __syncthreads();

    // --- Step 4: Compute Root-Mean-Square (RMS) over the 2 * head_dim features ---
    // Reduce across all threads using a shared memory layout or direct block parsing
    extern __shared__ float shared_mem[];
    float thread_sum_sq = 0.0f;

    for (int feat = tid; feat < v_dim; feat += num_threads) {
        float val = output[(i * num_heads + h) * v_dim + feat];
        thread_sum_sq += val * val;
    }
    shared_mem[tid] = thread_sum_sq;
    __syncthreads();

    // In-block tree reduction for thread_sum_sq
    for (int stride = num_threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    float total_sum_sq = shared_mem[0];
    float rms = sqrtf((1.0f / v_dim) * total_sum_sq + 1e-5f);

    // --- Step 5: Final Normalization and Scale Matrix Output Writing ---
    for (int feat = tid; feat < v_dim; feat += num_threads) {
        int target_idx = (i * num_heads + h) * v_dim + feat;
        if (total_sum_sq == 0.0f) {
            output[target_idx] = 0.0f;
        } else {
            output[target_idx] = (1.0f - lambda_init) * (output[target_idx] / rms);
        }
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, const float* lambdas,
                      float* output, int seq_len, int num_heads, int head_dim, float lambda_init) 
{
    // Execution configuration strategy: 
    // Grid maps along [Sequence Length, Attention Heads]
    dim3 grid(seq_len, num_heads);
    
    // Choose block dimensions fitting nicely into a warp-multiple while handling feature length
    int threads_per_block = 256; 
    size_t shared_mem_size = threads_per_block * sizeof(float);

    diff_attn_kernel<<<grid, threads_per_block, shared_mem_size>>>(
        Q, K, V, lambdas, output, seq_len, num_heads, head_dim, lambda_init
    );
    
    // Maintain stream alignment or explicit synchronization if downstream relies on immediate CPU reads
    cudaDeviceSynchronize();
}
