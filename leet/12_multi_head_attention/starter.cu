#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Each block processes one (head, query_row) pair.
// Threads within the block cooperate across the head dimension d_k.
__global__ void mha_online_softmax_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int N,
    int d_model,
    int h,
    int d_k,
    float scale
) {
    int head = blockIdx.x;
    int q_row = blockIdx.y;

    if (head >= h || q_row >= N) return;

    int tid = threadIdx.x;
    int col_offset = head * d_k;

    // Shared memory layout:
    // [0 ... d_k - 1]         : query vector Q[q_row, col_offset : col_offset + d_k]
    // [d_k ... d_k + 31]      : warp reduction scratchpad
    // [d_k + 32]              : broadcast scalar for dot product / weights
    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_reduce = smem + d_k;
    float* s_scalar = smem + d_k + 32;

    // 1. Load the query slice into shared memory and initialize output accumulator
    float q_val = 0.0f;
    float acc = 0.0f;
    if (tid < d_k) {
        q_val = Q[q_row * d_model + col_offset + tid];
        s_q[tid] = q_val;
    }
    __syncthreads();

    // Online softmax running statistics
    float m_i = -FLT_MAX; // Running maximum score
    float l_i = 0.0f;     // Running sum of exponentials

    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int num_warps = (blockDim.x + 31) >> 5;

    // 2. Iterate over all key/value rows j = 0 ... N - 1
    for (int j = 0; j < N; ++j) {
        int kv_base = j * d_model + col_offset;

        // Compute partial dot product Q[q_row] * K[j]
        float partial = (tid < d_k) ? (q_val * K[kv_base + tid]) : 0.0f;

        // Warp-level sum reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        }
        if (lane_id == 0) {
            s_reduce[warp_id] = partial;
        }
        __syncthreads();

        // First warp reduces across warps and updates online softmax stats
        if (warp_id == 0) {
            float sum = (lane_id < num_warps) ? s_reduce[lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane_id == 0) {
                float score = sum * scale;
                float m_new = fmaxf(m_i, score);
                float alpha = expf(m_i - m_new);
                float beta  = expf(score - m_new);

                l_i = l_i * alpha + beta;
                m_i = m_new;

                // Broadcast rescaling factor and current unnormalized weight
                s_scalar[0] = alpha;
                s_scalar[1] = beta;
            }
        }
        __syncthreads();

        // 3. Rescale running accumulator and add weighted V[j]
        if (tid < d_k) {
            float alpha = s_scalar[0];
            float beta  = s_scalar[1];
            acc = acc * alpha + beta * V[kv_base + tid];
        }
        __syncthreads();
    }

    // 4. Normalize by final exponential sum and write output
    if (warp_id == 0 && lane_id == 0) {
        s_scalar[0] = l_i;
    }
    __syncthreads();

    if (tid < d_k) {
        output[q_row * d_model + col_offset + tid] = acc / s_scalar[0];
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {
    int d_k = d_model / h;
    float scale = 1.0f / sqrtf(static_cast<float>(d_k));

    // Round block size up to nearest warp multiple (up to 1024, matching max d_model)
    int threads = ((d_k + 31) / 32) * 32;
    if (threads < 32) threads = 32;

    dim3 grid(h, N);
    size_t smem_bytes = (d_k + 34) * sizeof(float);

    mha_online_softmax_kernel<<<grid, threads, smem_bytes>>>(
        Q, K, V, output, N, d_model, h, d_k, scale
    );
}
