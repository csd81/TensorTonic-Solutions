#include <cuda_runtime.h>

// Maximum supported feature dimension d <= 128
constexpr int MAX_D = 128;

__global__ void kda_forward_kernel(const float* __restrict__ Q,
                                   const float* __restrict__ K,
                                   const float* __restrict__ V,
                                   const float* __restrict__ alpha,
                                   const float* __restrict__ beta,
                                   float* __restrict__ output,
                                   int seq_len, int d) {
    int b = blockIdx.x;
    int j = threadIdx.x;  // Each thread owns column j of S (for j < d)

    // Shared memory buffers for current timestep vectors
    __shared__ float s_q[MAX_D];
    __shared__ float s_k[MAX_D];
    __shared__ float s_v[MAX_D];
    __shared__ float s_alpha[MAX_D];
    __shared__ float s_beta;

    // Register-resident column j of the recurrent state matrix S_t
    float s_col[MAX_D];
    #pragma unroll
    for (int i = 0; i < MAX_D; ++i) {
        s_col[i] = 0.0f;
    }

    // Base pointers for batch b
    size_t seq_offset = static_cast<size_t>(b) * seq_len * d;
    size_t beta_offset = static_cast<size_t>(b) * seq_len;

    const float* Q_b = Q + seq_offset;
    const float* K_b = K + seq_offset;
    const float* V_b = V + seq_offset;
    const float* alpha_b = alpha + seq_offset;
    const float* beta_b = beta + beta_offset;
    float* out_b = output + seq_offset;

    for (int t = 0; t < seq_len; ++t) {
        size_t step_offset = static_cast<size_t>(t) * d;

        // Cooperatively load vectors for step t into shared memory
        if (j < d) {
            s_q[j] = Q_b[step_offset + j];
            s_k[j] = K_b[step_offset + j];
            s_v[j] = V_b[step_offset + j];
            s_alpha[j] = alpha_b[step_offset + j];
        }
        if (j == 0) {
            s_beta = beta_b[t];
        }
        __syncthreads();

        if (j < d) {
            // 1. Apply channel-wise decay: s_tilde = alpha_t * s_{t-1}
            // 2. Compute dot product: u_j = k_t^T * s_tilde
            float u_j = 0.0f;
            for (int i = 0; i < d; ++i) {
                float decayed = s_alpha[i] * s_col[i];
                s_col[i] = decayed;
                u_j = fmaf(s_k[i], decayed, u_j);
            }

            // 3. Delta rule update: s_t[:, j] = s_tilde + beta_t * (v_t[j] - u_j) * k_t
            // 4. Output readout: o_t[j] = q_t^T * s_t[:, j]
            float scale = s_beta * (s_v[j] - u_j);
            float o_j = 0.0f;
            for (int i = 0; i < d; ++i) {
                float updated = fmaf(scale, s_k[i], s_col[i]);
                s_col[i] = updated;
                o_j = fmaf(s_q[i], updated, o_j);
            }

            out_b[step_offset + j] = o_j;
        }
        __syncthreads();
    }
}

// Q, K, V, alpha, beta, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* alpha,
                      const float* beta, float* output, int batch, int seq_len, int d) {
    if (batch <= 0 || seq_len <= 0 || d <= 0) return;

    dim3 grid(batch);
    dim3 block(MAX_D);  // 128 threads per block covers d <= 128

    kda_forward_kernel<<<grid, block>>>(Q, K, V, alpha, beta, output, seq_len, d);
}
