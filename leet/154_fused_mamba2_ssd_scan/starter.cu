#include <cuda_runtime.h>
#include <cmath>

// Chunk size tuned to fit well within shared memory while covering max head_dim/state_dim (128)
constexpr int CHUNK_SIZE = 64;
constexpr int BLOCK_THREADS = 256;

__device__ __forceinline__ float softplus_f32(float x) {
    // Numerically stable softplus for x in [-30, 30]
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// Kernel parallelizes across (batch, num_heads). Each thread block processes an entire sequence
// for one (b, h) pair, keeping the (head_dim x state_dim) state completely in shared memory.
__global__ void mamba2_ssd_chunked_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dt,
    const float* __restrict__ dt_bias,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D,
    const float* __restrict__ z,
    float* __restrict__ y,
    int seq_len,
    int num_heads,
    int head_dim,
    int num_groups,
    int state_dim
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;

    const int heads_per_group = num_heads / num_groups;
    const int g = h / heads_per_group;

    const float A_h = A[h];
    const float dt_bias_h = dt_bias[h];
    const float D_h = D[h];

    // Base pointers for batch b
    const size_t x_batch_offset = (size_t)b * seq_len * num_heads * head_dim;
    const size_t dt_batch_offset = (size_t)b * seq_len * num_heads;
    const size_t bc_batch_offset = (size_t)b * seq_len * num_groups * state_dim;

    // Shared memory layout (Total ~145 KB if dynamically allocated, or ~44 KB static):
    // Since max head_dim = 128 and max state_dim = 128, state S is at most 128 * 128 * 4 = 64 KB.
    // We use dynamic shared memory to support up to 128x128 state effortlessly on modern GPUs.
    extern __shared__ float smem[];

    // Partition shared memory:
    // S:          head_dim * state_dim  (persistent recurrent state)
    // s_X:        CHUNK_SIZE * head_dim
    // s_B:        CHUNK_SIZE * state_dim (holds B, then B_tilde)
    // s_C:        CHUNK_SIZE * state_dim (holds C, then C_tilde)
    // s_M:        CHUNK_SIZE * CHUNK_SIZE
    // s_dt:       CHUNK_SIZE
    // s_P:        CHUNK_SIZE (cumulative log decay)
    float* S    = smem;
    float* s_X  = S + head_dim * state_dim;
    float* s_B  = s_X + CHUNK_SIZE * head_dim;
    float* s_C  = s_B + CHUNK_SIZE * state_dim;
    float* s_M  = s_C + CHUNK_SIZE * state_dim;
    float* s_dt = s_M + CHUNK_SIZE * CHUNK_SIZE;
    float* s_P  = s_dt + CHUNK_SIZE;

    // 1. Zero-initialize recurrent state S[head_dim][state_dim]
    const int state_elems = head_dim * state_dim;
    for (int idx = tid; idx < state_elems; idx += BLOCK_THREADS) {
        S[idx] = 0.0f;
    }
    __syncthreads();

    // 2. Iterate over sequence in chunks of CHUNK_SIZE
    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += CHUNK_SIZE) {
        const int chunk_len = min(CHUNK_SIZE, seq_len - chunk_start);

        // Step A: Load dt, compute delta_t and log_a_t = delta_t * A_h
        if (tid < chunk_len) {
            int t = chunk_start + tid;
            float raw_dt = dt[dt_batch_offset + (size_t)t * num_heads + h] + dt_bias_h;
            float delta = softplus_f32(raw_dt);
            s_dt[tid] = delta;
            s_P[tid] = delta * A_h; // Temporarily store individual log-decay a_t
        }
        __syncthreads();

        // Prefix sum for cumulative log-decay P_i = sum_{k=0}^i a_k (single thread is fast for <= 64)
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < chunk_len; ++i) {
                acc += s_P[i];
                s_P[i] = acc;
            }
        }
        __syncthreads();

        // Step B: Cooperatively load X, B, C tiles into shared memory
        const int x_elems = chunk_len * head_dim;
        for (int idx = tid; idx < x_elems; idx += BLOCK_THREADS) {
            int i = idx / head_dim;
            int d = idx % head_dim;
            int t = chunk_start + i;
            s_X[idx] = x[x_batch_offset + ((size_t)t * num_heads + h) * head_dim + d];
        }

        const int bc_elems = chunk_len * state_dim;
        for (int idx = tid; idx < bc_elems; idx += BLOCK_THREADS) {
            int i = idx / state_dim;
            int n = idx % state_dim;
            int t = chunk_start + i;
            size_t offset = bc_batch_offset + ((size_t)t * num_groups + g) * state_dim + n;
            s_B[idx] = B[offset];
            s_C[idx] = C[offset];
        }
        __syncthreads();

        // Step C: Compute lower-triangular intra-chunk matrix M[i][j] for 0 <= j <= i < chunk_len
        // M[i][j] = exp(P[i] - P[j]) * dt[j] * dot(C[i], B[j])
        const int m_elems = chunk_len * chunk_len;
        for (int idx = tid; idx < m_elems; idx += BLOCK_THREADS) {
            int i = idx / chunk_len;
            int j = idx % chunk_len;
            if (j <= i) {
                float dot = 0.0f;
                const float* c_row = &s_C[i * state_dim];
                const float* b_row = &s_B[j * state_dim];
                #pragma unroll 4
                for (int n = 0; n < state_dim; ++n) {
                    dot += c_row[n] * b_row[n];
                }
                float decay = expf(s_P[i] - s_P[j]);
                s_M[i * CHUNK_SIZE + j] = decay * s_dt[j] * dot;
            } else {
                s_M[i * CHUNK_SIZE + j] = 0.0f;
            }
        }
        __syncthreads();

        // Step D: Compute Output Y = M * X + C_tilde * S^T + D_h * X, then gate with silu(z)
        // Each thread handles one (i, d) output element at a time
        for (int idx = tid; idx < x_elems; idx += BLOCK_THREADS) {
            int i = idx / head_dim;
            int d = idx % head_dim;

            // 1. Intra-chunk contribution: sum_{j=0}^i M[i][j] * X[j][d]
            float y_intra = 0.0f;
            const float* m_row = &s_M[i * CHUNK_SIZE];
            for (int j = 0; j <= i; ++j) {
                y_intra += m_row[j] * s_X[j * head_dim + d];
            }

            // 2. Inter-chunk state readout: exp(P[i]) * sum_{n=0}^{state_dim-1} C[i][n] * S[d][n]
            float y_inter = 0.0f;
            const float* c_row = &s_C[i * state_dim];
            const float* s_row = &S[d * state_dim];
            #pragma unroll 4
            for (int n = 0; n < state_dim; ++n) {
                y_inter += c_row[n] * s_row[n];
            }
            y_inter *= expf(s_P[i]);

            // 3. Skip connection + SiLU gate + Write to global memory
            int t = chunk_start + i;
            size_t out_offset = x_batch_offset + ((size_t)t * num_heads + h) * head_dim + d;
            float x_val = s_X[idx];
            float z_val = z[out_offset];

            float val = (y_intra + y_inter + D_h * x_val) * silu_f32(z_val);
            y[out_offset] = val;
        }
        __syncthreads();

        // Step E: Update persistent state S for the next chunk
        // S_new[d][n] = exp(P[last]) * S[d][n] + sum_{j=0}^{chunk_len-1} X[j][d] * (exp(P[last] - P[j]) * dt[j] * B[j][n])
        const float P_last = s_P[chunk_len - 1];
        const float chunk_decay = expf(P_last);

        // Pre-scale B in shared memory into B_tilde[j][n] = exp(P_last - P[j]) * dt[j] * B[j][n]
        for (int idx = tid; idx < bc_elems; idx += BLOCK_THREADS) {
            int j = idx / state_dim;
            float scale = expf(P_last - s_P[j]) * s_dt[j];
            s_B[idx] *= scale;
        }
        __syncthreads();

        // Rank-chunk_len outer product update into S[d][n]
        for (int idx = tid; idx < state_elems; idx += BLOCK_THREADS) {
            int d = idx / state_dim;
            int n = idx % state_dim;

            float acc = 0.0f;
            for (int j = 0; j < chunk_len; ++j) {
                acc += s_X[j * head_dim + d] * s_B[j * state_dim + n];
            }
            S[idx] = chunk_decay * S[idx] + acc;
        }
        __syncthreads();
    }
}

extern "C" void solve(const float* x, const float* dt, const float* dt_bias, const float* A,
                      const float* B, const float* C, const float* D, const float* z, float* y,
                      int batch, int seq_len, int num_heads, int head_dim, int num_groups,
                      int state_dim) {
    if (batch <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0 || state_dim <= 0) {
        return;
    }

    dim3 grid(batch, num_heads);
    dim3 block(BLOCK_THREADS);

    // Compute exact dynamic shared memory needed for max dimensions
    size_t smem_floats = (size_t)head_dim * state_dim        // S
                       + (size_t)CHUNK_SIZE * head_dim       // s_X
                       + (size_t)CHUNK_SIZE * state_dim      // s_B
                       + (size_t)CHUNK_SIZE * state_dim      // s_C
                       + (size_t)CHUNK_SIZE * CHUNK_SIZE     // s_M
                       + (size_t)CHUNK_SIZE                  // s_dt
                       + (size_t)CHUNK_SIZE;                 // s_P
    size_t smem_bytes = smem_floats * sizeof(float);

    // Request >48KB shared memory opt-in (required for 128x128 state = 64KB + tiles = ~145KB)
    cudaFuncSetAttribute(
        mamba2_ssd_chunked_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)
    );

    mamba2_ssd_chunked_kernel<<<grid, block, smem_bytes>>>(
        x, dt, dt_bias, A, B, C, D, z, y,
        seq_len, num_heads, head_dim, num_groups, state_dim
    );
}
