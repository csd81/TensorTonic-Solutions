#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define BLOCK_M 32
#define BLOCK_N 32
#define MAX_HEAD_DIM 128

__global__ void sliding_window_gqa_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int window_size,
    float softcap)
{
    int h = blockIdx.y;
    int q_block_idx = blockIdx.x;
    int tid = threadIdx.x; // 0 .. BLOCK_M - 1

    int q_idx = q_block_idx * BLOCK_M + tid;
    int group_size = num_q_heads / num_kv_heads;
    int g = h / group_size;

    const float* Q_head = Q + (size_t)h * seq_len * head_dim;
    const float* K_head = K + (size_t)g * seq_len * head_dim;
    const float* V_head = V + (size_t)g * seq_len * head_dim;
    float* O_head       = output + (size_t)h * seq_len * head_dim;

    // Load this thread's query vector into registers
    float q_reg[MAX_HEAD_DIM];
    float o_reg[MAX_HEAD_DIM];
    bool valid_q = (q_idx < seq_len);

    if (valid_q) {
        for (int d = 0; d < head_dim; ++d) {
            q_reg[d] = Q_head[q_idx * head_dim + d];
            o_reg[d] = 0.0f;
        }
    }

    // Online softmax state
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    float inv_sqrt_d_softcap = 1.0f / (sqrtf((float)head_dim) * softcap);

    // Determine range of key indices needed across the entire query block
    int block_q_start = q_block_idx * BLOCK_M;
    int block_q_end   = min(seq_len - 1, block_q_start + BLOCK_M - 1);

    int min_k = max(0, block_q_start - window_size + 1);
    int max_k = block_q_end;

    int start_tile = min_k / BLOCK_N;
    int end_tile   = max_k / BLOCK_N;

    // Shared memory buffers for K and V tiles
    __shared__ float K_tile[BLOCK_N][MAX_HEAD_DIM];
    __shared__ float V_tile[BLOCK_N][MAX_HEAD_DIM];

    int q_win_start = valid_q ? max(0, q_idx - window_size + 1) : 0;

    for (int tile = start_tile; tile <= end_tile; ++tile) {
        int k_tile_start = tile * BLOCK_N;

        // Cooperatively load K and V tiles into shared memory
        // Each of the BLOCK_M (32) threads loads 1 row of BLOCK_N (32)
        int k_load_idx = k_tile_start + tid;
        if (k_load_idx < seq_len) {
            for (int d = 0; d < head_dim; ++d) {
                K_tile[tid][d] = K_head[k_load_idx * head_dim + d];
                V_tile[tid][d] = V_head[k_load_idx * head_dim + d];
            }
        } else {
            for (int d = 0; d < head_dim; ++d) {
                K_tile[tid][d] = 0.0f;
                V_tile[tid][d] = 0.0f;
            }
        }
        __syncthreads();

        if (valid_q) {
            int tile_limit = min(BLOCK_N, seq_len - k_tile_start);

            for (int j = 0; j < tile_limit; ++j) {
                int k_idx = k_tile_start + j;

                // Enforce causal + sliding window constraints
                if (k_idx >= q_win_start && k_idx <= q_idx) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_reg[d] * K_tile[j][d];
                    }

                    // Apply scaling and tanh soft-capping
                    float score = softcap * tanhf(dot * inv_sqrt_d_softcap);

                    // Online softmax update
                    float m_new = fmaxf(m_i, score);
                    float alpha = expf(m_i - m_new);
                    float beta  = expf(score - m_new);

                    l_i = l_i * alpha + beta;

                    for (int d = 0; d < head_dim; ++d) {
                        o_reg[d] = o_reg[d] * alpha + beta * V_tile[j][d];
                    }

                    m_i = m_new;
                }
            }
        }
        __syncthreads();
    }

    // Normalize and write final output
    if (valid_q) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            O_head[q_idx * head_dim + d] = o_reg[d] * inv_l;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim, int window_size,
                      float softcap)
{
    dim3 block(BLOCK_M);
    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, num_q_heads);

    sliding_window_gqa_kernel<<<grid, block>>>(
        Q, K, V, output,
        num_q_heads, num_kv_heads, seq_len, head_dim, window_size, softcap
    );
}
