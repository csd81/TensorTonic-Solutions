#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

constexpr int BLOCK_N = 64;   // Key/Value tile size along seq_len
constexpr int MAX_HEAD_DIM = 128;

__global__ void fused_gqa_rope_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim
) {
    // Grid: (seq_len, num_q_heads) -> Each block computes 1 query token for 1 query head
    const int q_pos = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x; // 0 .. BLOCK_N - 1 (64 threads per block)

    if (q_pos >= seq_len || q_head >= num_q_heads) return;

    const int group_size = num_q_heads / num_kv_heads;
    const int kv_head = q_head / group_size;
    const int half_dim = head_dim >> 1;
    const float scale = rsqrtf((float)head_dim);

    // Pointers to current head's data
    const float* q_ptr = Q + ((size_t)q_head * seq_len + q_pos) * head_dim;
    const float* k_base = K + ((size_t)kv_head * seq_len) * head_dim;
    const float* v_base = V + ((size_t)kv_head * seq_len) * head_dim;
    float* out_ptr = output + ((size_t)q_head * seq_len + q_pos) * head_dim;

    // Shared memory allocations
    __shared__ float s_q[MAX_HEAD_DIM];
    __shared__ float s_k[BLOCK_N][MAX_HEAD_DIM];
    __shared__ float s_v[BLOCK_N][MAX_HEAD_DIM];
    __shared__ float s_scores[BLOCK_N];
    __shared__ float s_reduce[BLOCK_N];
    __shared__ float s_out[MAX_HEAD_DIM];

    // 1. Load Q[q_head, q_pos] into shared memory and apply RoPE
    if (tid < half_dim) {
        const int table_idx = q_pos * half_dim + tid;
        const float c = cos_table[table_idx];
        const float s = sin_table[table_idx];
        const float q0 = q_ptr[tid];
        const float q1 = q_ptr[tid + half_dim];

        s_q[tid] = q0 * c - q1 * s;
        s_q[tid + half_dim] = q1 * c + q0 * s;
    }

    // Initialize shared output accumulator
    for (int d = tid; d < head_dim; d += BLOCK_N) {
        s_out[d] = 0.0f;
    }
    __syncthreads();

    // Online softmax running statistics
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    // 2. Iterate over causal Key/Value tiles up to q_pos
    const int num_tiles = (q_pos + BLOCK_N) / BLOCK_N;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_start = tile * BLOCK_N;
        const int kv_pos = tile_start + tid;
        const bool valid_kv = (kv_pos <= q_pos);

        // Cooperatively load K (with RoPE fused) and V for this tile into shared memory
        // Total pairs to load across tile = BLOCK_N * half_dim
        const int total_pairs = BLOCK_N * half_dim;
        for (int idx = tid; idx < total_pairs; idx += BLOCK_N) {
            const int r = idx / half_dim; // row in tile (0 .. BLOCK_N - 1)
            const int d = idx % half_dim; // pair index (0 .. half_dim - 1)
            const int cur_kv_pos = tile_start + r;

            if (cur_kv_pos <= q_pos) {
                const size_t kv_offset = (size_t)cur_kv_pos * head_dim;
                const int table_idx = cur_kv_pos * half_dim + d;
                const float c = cos_table[table_idx];
                const float s = sin_table[table_idx];

                const float k0 = k_base[kv_offset + d];
                const float k1 = k_base[kv_offset + d + half_dim];

                s_k[r][d] = k0 * c - k1 * s;
                s_k[r][d + half_dim] = k1 * c + k0 * s;

                s_v[r][d] = v_base[kv_offset + d];
                s_v[r][d + half_dim] = v_base[kv_offset + d + half_dim];
            }
        }
        __syncthreads();

        // 3. Each thread computes dot product Q[q_pos] . K[tile_start + tid]
        float score = -FLT_MAX;
        if (valid_kv) {
            float dot = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < head_dim; ++d) {
                dot += s_q[d] * s_k[tid][d];
            }
            score = dot * scale;
        }
        s_scores[tid] = score;
        s_reduce[tid] = score;
        __syncthreads();

        // 4. Block-wide reduction to find tile maximum score
        for (int stride = BLOCK_N >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
            }
            __syncthreads();
        }
        const float m_tile = s_reduce[0];
        const float m_new = fmaxf(m_prev, m_tile);

        // 5. Compute unnormalized attention weights exp(score - m_new)
        float p = valid_kv ? expf(s_scores[tid] - m_new) : 0.0f;
        s_scores[tid] = p;
        s_reduce[tid] = p;
        __syncthreads();

        // 6. Block-wide reduction to find sum of weights in tile
        for (int stride = BLOCK_N >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduce[tid] += s_reduce[tid + stride];
            }
            __syncthreads();
        }
        const float l_tile = s_reduce[0];

        // Rescale factor for previous accumulators
        const float alpha = (l_prev == 0.0f) ? 0.0f : expf(m_prev - m_new);
        const float l_new = l_prev * alpha + l_tile;

        // 7. Update output accumulator across head_dim
        for (int d = tid; d < head_dim; d += BLOCK_N) {
            float pv_sum = 0.0f;
            #pragma unroll 8
            for (int r = 0; r < BLOCK_N; ++r) {
                // s_scores[r] is 0.0f for masked/out-of-bounds positions
                pv_sum += s_scores[r] * s_v[r][d];
            }
            s_out[d] = s_out[d] * alpha + pv_sum;
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    // 8. Final normalization and write to global output
    const float inv_l = 1.0f / l_prev;
    for (int d = tid; d < head_dim; d += BLOCK_N) {
        out_ptr[d] = s_out[d] * inv_l;
    }
}

// Q, K, V, cos, sin, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* cos,
                      const float* sin, float* output, int num_q_heads, int num_kv_heads,
                      int seq_len, int head_dim) {
    if (seq_len <= 0 || num_q_heads <= 0) return;

    dim3 grid(seq_len, num_q_heads);
    dim3 block(BLOCK_N);

    fused_gqa_rope_kernel<<<grid, block>>>(
        Q, K, V, cos, sin, output,
        num_q_heads, num_kv_heads, seq_len, head_dim
    );
}
