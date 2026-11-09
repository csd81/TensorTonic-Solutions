
#include <cuda_runtime.h>
#include <cmath>

#define FLOAT4(pointer) (reinterpret_cast<const float4*>(pointer))
#define FLOAT4_MUT(pointer) (reinterpret_cast<float4*>(pointer))

// ----------------------------------------------------------------------------
// Weight Buffer Offsets
// ----------------------------------------------------------------------------
constexpr int OFFSET_GAMMA1   = 0;
constexpr int OFFSET_BETA1    = 768;
constexpr int OFFSET_WQKV     = 1536;
constexpr int OFFSET_BQKV     = 1771008;
constexpr int OFFSET_WATTN    = 1773312;
constexpr int OFFSET_BATTN    = 2363136;
constexpr int OFFSET_GAMMA2   = 2363904;
constexpr int OFFSET_BETA2    = 2364672;
constexpr int OFFSET_WFC      = 2365440;
constexpr int OFFSET_BFC      = 4724736;
constexpr int OFFSET_WPROJ    = 4727808;
constexpr int OFFSET_BPROJ    = 7087104;

// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------

__global__ void layernorm1_qkv_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weights,
    float* __restrict__ qkv_out,
    int seq_len
) {
    // 1 block handles 1 row (token) of the sequence
    int r = blockIdx.x;
    if (r >= seq_len) return;

    const float* gamma1 = weights + OFFSET_GAMMA1;
    const float* beta1 = weights + OFFSET_BETA1;
    const float* w_qkv = weights + OFFSET_WQKV;
    const float* b_qkv = weights + OFFSET_BQKV;

    const float* x_row = x + r * 768;
    float* qkv_row = qkv_out + r * 2304;

    // Step 1: Compute mean and variance using shared memory reduction
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ float s_buf[256];

    int tid = threadIdx.x;
    
    // Each thread processes 3 elements for d_model=768
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        sum += x_row[tid + i * 256];
    }
    s_buf[tid] = sum;
    __syncthreads();

    // Reduction for mean
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s_buf[tid] += s_buf[tid + stride];
        __syncthreads();
    }
    if (tid == 0) s_mean = s_buf[0] / 768.0f;
    __syncthreads();

    float mean = s_mean;
    float sq_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        float diff = x_row[tid + i * 256] - mean;
        sq_sum += diff * diff;
    }
    s_buf[tid] = sq_sum;
    __syncthreads();

    // Reduction for variance
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s_buf[tid] += s_buf[tid + stride];
        __syncthreads();
    }
    if (tid == 0) s_buf[0] = 1.0f / sqrtf((s_buf[0] / 768.0f) + 1e-5f);
    __syncthreads();
    float inv_std = s_buf[0];

    // Step 2: Compute LN1 dynamic outputs locally and stream into QKV GEMV
    // Pre-load the token row normalized into local registers
    float x_norm[3];
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        int idx = tid + i * 256;
        x_norm[i] = ((x_row[idx] - mean) * inv_std) * gamma1[idx] + beta1[idx];
    }

    // Matrix Multiplication for QKV: (1, 768) x (768, 2304) -> (1, 2304)
    // 2304 / 4 float4 items = 576 total outputs vector-wide.
    // Each thread among 256 calculates around 2 to 3 float4 chunks.
    for (int out_idx = tid; out_idx < 576; out_idx += 256) {
        float4 accum = FLOAT4(b_qkv)[out_idx];
        int col_base = out_idx * 4;

        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            int r_idx = threadIdx.x + i * 256;
            const float* w_ptr = w_qkv + r_idx * 2304 + col_base;
            float x_val = x_norm[i];
            accum.x += x_val * w_ptr[0];
            accum.y += x_val * w_ptr[1];
            accum.z += x_val * w_ptr[2];
            accum.w += x_val * w_ptr[3];
        }

        // Intra-block shuffle reduction across threads matching same output columns
        // Standard warp/block reduction isn't fully required here if loops are processed sequentially,
        // but since threads share the workload of row elements, we accumulate over the blocks via atomic or separate reduction.
        // Instead, to maximize parallel memory coalescing, we swap loops:
    }
    __syncthreads();

    // Clear shared memory for accumulation mapping
    // Alternatively, standard approach: map threads directly to output elements.
    for (int out_col = tid; out_col < 2304; out_col += 256) {
        float accum = b_qkv[out_col];
        for (int k = 0; k < 768; ++k) {
            // Compute normalized value on-the-fly to save local memory pressure
            float x_n = ((x_row[k] - mean) * inv_std) * gamma1[k] + beta1[k];
            accum += x_n * w_qkv[k * 2304 + out_col];
        }
        qkv_row[out_col] = accum;
    }
}

__global__ void multi_head_attention_kernel(
    const float* __restrict__ qkv,
    float* __restrict__ mha_out,
    int seq_len
) {
    // Computes per-head scaled dot-product attention without causal mask
    // Thread block maps to (Sequence Token row, Head Index)
    int r = blockIdx.x; // row index inside sequence
    int h = blockIdx.y; // head index [0, 11]
    
    if (r >= seq_len || h >= 12) return;

    int tid = threadIdx.x;

    // Dynamic allocation mapping for softmax scores
    // Up to seq_len max elements. For large scales, global tracking might be needed,
    // but with 1024 baseline performance target, we store in shared or compute stream-based.
    extern __shared__ float s_scores[];

    // Locating Head pointers
    // QKV structure is packed as [seq_len, 3, 12, 64]
    const float* q_ptr = qkv + r * 2304 + (0 * 768) + h * 64;
    
    // Scale factor: 1 / sqrt(64) = 0.125f
    const float scale = 0.125f;

    // Step 1: Compute Attention Scores = Q * K^T
    for (int k_idx = tid; k_idx < seq_len; k_idx += 256) {
        const float* k_ptr = qkv + k_idx * 2304 + (1 * 768) + h * 64;
        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < 64; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        s_scores[k_idx] = score * scale;
    }
    __syncthreads();

    // Step 2: Safe Softmax on s_scores [0 ... seq_len - 1]
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_reduce_buf[256];

    // Find Max
    float local_max = -1e20f;
    for (int i = tid; i < seq_len; i += 256) {
        if (s_scores[i] > local_max) local_max = s_scores[i];
    }
    s_reduce_buf[tid] = local_max;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_reduce_buf[tid + stride] > s_reduce_buf[tid]) {
                s_reduce_buf[tid] = s_reduce_buf[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) s_max = s_reduce_buf[0];
    __syncthreads();

    // Compute exp and Sum
    float local_sum = 0.0f;
    float max_val = s_max;
    for (int i = tid; i < seq_len; i += 256) {
        float e = expf(s_scores[i] - max_val);
        s_scores[i] = e;
        local_sum += e;
    }
    s_reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s_reduce_buf[tid] += s_reduce_buf[tid + stride];
        __syncthreads();
    }
    if (tid == 0) s_sum = s_reduce_buf[0];
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / s_sum;
    for (int i = tid; i < seq_len; i += 256) {
        s_scores[i] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Compute Attention Output = Scores * V
    // Output shape for this head is 64 channels
    float* out_ptr = mha_out + r * 768 + h * 64;
    for (int d = tid; d < 64; d += 256) {
        float out_val = 0.0f;
        for (int v_idx = 0; v_idx < seq_len; ++v_idx) {
            const float* v_ptr = qkv + v_idx * 2304 + (2 * 768) + h * 64;
            out_val += s_scores[v_idx] * v_ptr[d];
        }
        out_ptr[d] = out_val;
    }
}

__global__ void attn_proj_residual_ln2_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mha_out,
    const float* __restrict__ weights,
    float* __restrict__ ln2_out,
    float* __restrict__ x_prime_out,
    int seq_len
) {
    int r = blockIdx.x;
    if (r >= seq_len) return;

    int tid = threadIdx.x;

    const float* w_attn = weights + OFFSET_WATTN;
    const float* b_attn = weights + OFFSET_BATTN;
    const float* gamma2 = weights + OFFSET_GAMMA2;
    const float* beta2 = weights + OFFSET_BETA2;

    const float* mha_row = mha_out + r * 768;
    const float* x_row = x + r * 768;
    float* x_prime_row = x_prime_out + r * 768;
    float* ln2_row = ln2_out + r * 768;

    // Step 1: Linear Attention Output Projection + Add Residual 1 on the fly
    // Compute whole row of x_prime
    __shared__ float s_x_prime[768];

    for (int col = tid; col < 768; col += 256) {
        float accum = b_attn[col];
        #pragma unroll(4)
        for (int k = 0; k < 768; ++k) {
            accum += mha_row[k] * w_attn[k * 768 + col];
        }
        float xp = x_row[col] + accum;
        s_x_prime[col] = xp;
        x_prime_row[col] = xp; // Save globally for the final residual 2 step
    }
    __syncthreads();

    // Step 2: Layer Norm 2 across s_x_prime
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ float s_buf[256];

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        sum += s_x_prime[tid + i * 256];
    }
    s_buf[tid] = sum;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s_buf[tid] += s_buf[tid + stride];
        __syncthreads();
    }
    if (tid == 0) s_mean = s_buf[0] / 768.0f;
    __syncthreads();

    float mean = s_mean;
    float sq_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        float diff = s_x_prime[tid + i * 256] - mean;
        sq_sum += diff * diff;
    }
    s_buf[tid] = sq_sum;
    __syncthreads();


for (int stride = 128; stride > 0; stride >>= 1) {
if (tid < stride) s_buf[tid] += s_buf[tid + stride];
__syncthreads();
}
if (tid == 0) s_buf[0] = 1.0f / sqrtf((s_buf[0] / 768.0f) + 1e-5f);
__syncthreads();
float inv_std = s_buf[0];
// Compute normalized output and write to global memory
for (int col = tid; col < 768; col += 256) {
ln2_row[col] = ((s_x_prime[col] - mean) * inv_std) * gamma2[col] + beta2[col];
}
}
inline device float gelu_tanh(float x) {
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
return 0.5f * x * (1.0f + tanhf(inner));
}
global void ffn_fused_kernel(
const float* restrict ln2_out,
const float* restrict x_prime,
const float* restrict weights,
float* restrict output,
int seq_len
) {
int r = blockIdx.x;
if (r >= seq_len) return;
int tid = threadIdx.x;
const float* w_fc = weights + OFFSET_WFC;
const float* b_fc = weights + OFFSET_BFC;
const float* w_proj = weights + OFFSET_WPROJ;
const float* b_proj = weights + OFFSET_BPROJ;
const float* ln2_row = ln2_out + r * 768;
const float* x_prime_row = x_prime + r * 768;
float* out_row = output + r * 768;
// Since ffn_dim = 3072 is quite large, dynamically compute intermediate values
// to map linear operations directly back to output 768 shape dimension.
// We break projection down into thread loops.
for (int col = tid; col < 768; col += 256) {
float accum_proj = b_proj[col];
// Loop over 3072 dimensions of hidden size
for (int k = 0; k < 3072; ++k) {
// Compute the k-th item of the GELU activation layer inline
float intermediate = b_fc[k];
#pragma unroll(4)
for (int d = 0; d < 768; ++d) {
intermediate += ln2_row[d] * w_fc[d * 3072 + k];
}
float act = gelu_tanh(intermediate);
accum_proj += act * w_proj[k * 768 + col];
}
// Step 8: Residual 2 Addition
out_row[col] = x_prime_row[col] + accum_proj;
}
}
// ----------------------------------------------------------------------------
// Solver Entry Point
// ----------------------------------------------------------------------------
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {
// Allocate required temporary buffers on the GPU device dynamically
float* d_qkv;
float* d_mha_out;
float* d_ln2_out;
float* d_x_prime;
cudaMalloc(&d_qkv, seq_len * 2304 * sizeof(float));
cudaMalloc(&d_mha_out, seq_len * 768 * sizeof(float));
cudaMalloc(&d_ln2_out, seq_len * 768 * sizeof(float));
cudaMalloc(&d_x_prime, seq_len * 768 * sizeof(float));
// Execution Configurations
dim3 grid_mha(seq_len, 12);
size_t shared_mha_size = seq_len * sizeof(float);
// 1. Layer Norm 1 & QKV Projection
layernorm1_qkv_fused_kernel<<<seq_len, 256>>>(x, weights, d_qkv, seq_len);
// 2. Multi-Head Attention Mechanism
multi_head_attention_kernel<<<grid_mha, 256, shared_mha_size>>>(d_qkv, d_mha_out, seq_len);
// 3. Output Projection & Layer Norm 2 (Fused Residual 1)
attn_proj_residual_ln2_kernel<<<seq_len, 256>>>(x, d_mha_out, weights, d_ln2_out, d_x_prime, seq_len);
// 4. Feed-Forward Network & Residual 2 Block Output
ffn_fused_kernel<<<seq_len, 256>>>(d_ln2_out, d_x_prime, weights, output, seq_len);
// Synchronization and Cleanup
cudaDeviceSynchronize();
cudaFree(d_qkv);
cudaFree(d_mha_out);
cudaFree(d_ln2_out);
cudaFree(d_x_prime);
}
