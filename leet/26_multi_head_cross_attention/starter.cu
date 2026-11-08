#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Warp-level sum reduction
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel: One warp (32 threads) processes one query vector Q[m, h, :]
// Across all N keys/values using online softmax (no N-sized global memory workspace required).
__global__ void cross_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M, int N, int H, int D,
    float scale
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    int total_queries = M * H;
    if (warp_id >= total_queries) return;

    int m = warp_id / H;
    int h = warp_id % H;

    // Max D is 256; with 32 threads per warp, each thread handles at most 8 elements
    float q_reg[8];
    float acc_o[8];

    int q_base = (m * H + h) * D;

    // Load Q[m, h, :] into registers and initialize output accumulators
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int d = lane_id + i * 32;
        if (d < D) {
            q_reg[i] = Q[q_base + d];
        } else {
            q_reg[i] = 0.0f;
        }
        acc_o[i] = 0.0f;
    }

    // Online softmax running statistics
    float m_max = -FLT_MAX;
    float l_sum = 0.0f;

    // Iterate over all N key/value positions
    for (int n = 0; n < N; ++n) {
        int kv_base = (n * H + h) * D;

        // 1. Compute dot product Q[m, h, :] . K[n, h, :]
        float partial_dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int d = lane_id + i * 32;
            if (d < D) {
                partial_dot += q_reg[i] * K[kv_base + d];
            }
        }

        // Reduce dot product across the 32 lanes of the warp and broadcast to all lanes
        float dot = warpReduceSum(partial_dot);
        dot = __shfl_sync(0xffffffff, dot, 0);

        float score = dot * scale;

        // 2. Update online softmax statistics
        float m_new = fmaxf(m_max, score);
        float alpha = expf(m_max - m_new);
        float beta  = expf(score - m_new);

        l_sum = l_sum * alpha + beta;
        m_max = m_new;

        // 3. Rescale running output and accumulate weighted V[n, h, :]
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int d = lane_id + i * 32;
            if (d < D) {
                acc_o[i] = acc_o[i] * alpha + beta * V[kv_base + d];
            }
        }
    }

    // 4. Final normalization and write to global memory
    float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int d = lane_id + i * 32;
        if (d < D) {
            output[q_base + d] = acc_o[i] * inv_l;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int H, int D) {
    int total_queries = M * H;
    if (total_queries <= 0 || N <= 0 || D <= 0) return;

    float scale = 1.0f / sqrtf((float)D);

    // 4 warps (128 threads) per block
    int threads_per_block = 128;
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (total_queries + warps_per_block - 1) / warps_per_block;

    cross_attention_kernel<<<num_blocks, threads_per_block>>>(
        Q, K, V, output, M, N, H, D, scale
    );
}
