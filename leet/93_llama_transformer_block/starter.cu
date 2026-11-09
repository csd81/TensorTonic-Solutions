
#include <cuda_runtime.h>
#include <math.h>

#define D_MODEL 512
#define HEAD_DIM 64
#define N_Q_HEADS 8
#define N_KV_HEADS 2
#define FFN_HIDDEN 1408
#define GQA_GROUP 4

// Offsets in the packed weights buffer
#define OFF_W1 0
#define OFF_WQ 512
#define OFF_WK 262656
#define OFF_WV 328192
#define OFF_WO 393728
#define OFF_W2 655872
#define OFF_WGATE 656384
#define OFF_WUP 1377280
#define OFF_WDOWN 2098176

// -----------------------------------------------------------------------------
// CUDA Kernels
// -----------------------------------------------------------------------------

// Computes RMSNorm on rows of 512 elements and performs the QKV linear projections.
__global__ void rmsnorm_qkv_kernel(const float* __restrict__ x,
                                   const float* __restrict__ w1,
                                   const float* __restrict__ W_Q,
                                   const float* __restrict__ W_K,
                                   const float* __restrict__ W_V,
                                   float* __restrict__ Q,
                                   float* __restrict__ K,
                                   float* __restrict__ V,
                                   int seq_len) {
    int t = blockIdx.x; // sequence token index
    if (t >= seq_len) return;

    int tid = threadIdx.x; // 0 to 511

    // Step 1: Compute variance for RMSNorm
    float val = x[t * D_MODEL + tid];
    float sq_val = val * val;

    // Parallel reduction within warp/block
    for (int offset = 16; offset > 0; offset /= 2) {
        sq_val += __shfl_down_sync(0xffffffff, sq_val, offset);
    }
    __shared__ float s_variance;
    if ((tid & 31) == 0) {
        atomicAdd(&s_variance, sq_val);
    }
    __syncthreads();

    float inv_rms = rsqrtf(s_variance / (float)D_MODEL + 1e-5f);
    float norm_x = val * inv_rms * w1[tid];

    // Shared memory to hold normalized row for GEMV projections
    __shared__ float s_norm_x[D_MODEL];
    s_norm_x[tid] = norm_x;
    __syncthreads();

    // Step 2: Q Projection (512 heads * 64 dim = 512 total outputs)
    // Map threadIdx.x directly to output dim
    float q_acc = 0.0f;
    for (int i = 0; i < D_MODEL; ++i) {
        q_acc += s_norm_x[i] * W_Q[tid * D_MODEL + i];
    }
    Q[t * D_MODEL + tid] = q_acc;

    // Step 3: K and V Projections (128 outputs each)
    if (tid < 128) {
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        for (int i = 0; i < D_MODEL; ++i) {
            float nx = s_norm_x[i];
            k_acc += nx * W_K[tid * D_MODEL + i];
            v_acc += nx * W_V[tid * D_MODEL + i];
        }
        K[t * 128 + tid] = k_acc;
        V[t * 128 + tid] = v_acc;
    }
}

// Applies RoPE to both Q and K heads independently
__global__ void apply_rope_kernel(float* __restrict__ Q,
                                  float* __restrict__ K,
                                  const float* __restrict__ cos_tab,
                                  const float* __restrict__ sin_tab,
                                  int seq_len) {
    int t = blockIdx.x; // token position
    int h = blockIdx.y; // head index
    int i = threadIdx.x; // index within half-dim (0 to 31)

    if (t >= seq_len) return;

    int half_dim = HEAD_DIM / 2;
    float c = cos_tab[t * half_dim + i];
    float s = sin_tab[t * half_dim + i];

    if (h < N_Q_HEADS) {
        int q_idx1 = t * (N_Q_HEADS * HEAD_DIM) + h * HEAD_DIM + i;
        int q_idx2 = q_idx1 + half_dim;
        float q1 = Q[q_idx1];
        float q2 = Q[q_idx2];
        Q[q_idx1] = q1 * c - q2 * s;
        Q[q_idx2] = q1 * s + q2 * c;
    }

    if (h < N_KV_HEADS) {
        int k_idx1 = t * (N_KV_HEADS * HEAD_DIM) + h * HEAD_DIM + i;
        int k_idx2 = k_idx1 + half_dim;
        float k1 = K[k_idx1];
        float h2 = K[k_idx2];
        K[k_idx1] = k1 * c - h2 * s;
        K[k_idx2] = k1 * s + h2 * c;
    }
}

// Causal attention execution including Grouped Query Attention (GQA) mapping
__global__ void attention_kernel(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 float* __restrict__ O,
                                 int seq_len) {
    int t = blockIdx.x; // Target token query row
    int h = blockIdx.y; // Q head index (0 to 7)
    int tid = threadIdx.x; // Lane index (0 to 63) for output accumulation

    if (t >= seq_len) return;

    int kv_h = h / GQA_GROUP; // GQA head mapping

    extern __shared__ float s_scores[]; // Dynamic shared memory for row-softmax

    // Step 1: Compute Attention Scores with Causal Masking
    float max_val = -1e20f;
    for (int j = 0; j <= t; ++j) {
        float score = 0.0f;
        // Compute dot product of Q[t, h] and K[j, kv_h]
        for (int d = 0; d < HEAD_DIM; ++d) {
            score += Q[t * (N_Q_HEADS * HEAD_DIM) + h * HEAD_DIM + d] *
                     K[j * (N_KV_HEADS * HEAD_DIM) + kv_h * HEAD_DIM + d];
        }
        score /= 8.0f; // sqrt(64)
        s_scores[j] = score;
        if (score > max_val) max_val = score;
    }

    // Step 2: Stable Softmax
    float sum_exp = 0.0f;
    for (int j = 0; j <= t; ++j) {
        s_scores[j] = expf(s_scores[j] - max_val);
        sum_exp += s_scores[j];
    }
    for (int j = 0; j <= t; ++j) {
        s_scores[j] /= sum_exp;
    }

    // Step 3: Weighted sum over V values
    float out_val = 0.0f;
    for (int j = 0; j <= t; ++j) {
        out_val += s_scores[j] * V[j * (N_KV_HEADS * HEAD_DIM) + kv_h * HEAD_DIM + tid];
    }

    // Write directly out to row-major intermediate output tensor
    O[t * (N_Q_HEADS * HEAD_DIM) + h * HEAD_DIM + tid] = out_val;
}

// Out-projection and First Residual accumulation
__global__ void out_proj_residual_kernel(const float* __restrict__ attn_out,
                                         const float* __restrict__ W_O,
                                         const float* __restrict__ x,
                                         float* __restrict__ x_prime,
                                         int seq_len) {
    int t = blockIdx.x;
    int tid = threadIdx.x; // 0 to 511

    if (t >= seq_len) return;

    __shared__ float s_attn[D_MODEL];
    s_attn[tid] = attn_out[t * D_MODEL + tid];
    __syncthreads();

    float acc = 0.0f;
    for (int i = 0; i < D_MODEL; ++i) {
        acc += s_attn[i] * W_O[tid * D_MODEL + i];
    }

    x_prime[t * D_MODEL + tid] = x[t * D_MODEL + tid] + acc;
}

// RMSNorm 2 followed by SwiGLU FFN (Gate + Up Projection + Activation)
__global__ void ffn_gate_up_kernel(const float* __restrict__ x_prime,
                                   const float* __restrict__ w2,
                                   const float* __restrict__ W_gate,
                                   const float* __restrict__ W_up,
                                   float* __restrict__ intermediate_ffn,
                                   int seq_len) {
    int t = blockIdx.x;
    int tid = threadIdx.x; // 0 to 511

    if (t >= seq_len) return;

    float val = x_prime[t * D_MODEL + tid];
    float sq_val = val * val;

    for (int offset = 16; offset > 0; offset /= 2) {
        sq_val += __shfl_down_sync(0xffffffff, sq_val, offset);
    }
    __shared__ float s_variance;
    if ((tid & 31) == 0) {
        atomicAdd(&s_variance, sq_val);
    }
    __syncthreads();

    float inv_rms = rsqrtf(s_variance / (float)D_MODEL + 1e-5f);
    float norm_x = val * inv_rms * w2[tid];

    __shared__ float s_norm_x[D_MODEL];
    s_norm_x[tid] = norm_x;
    __syncthreads();

    // Compute chunks of the 1408 FFN Hidden Dimension
    // Each thread handles 2-3 hidden channels sequentially
    for (int h = tid; h < FFN_HIDDEN; h += D_MODEL) {
        float g_acc = 0.0f;
        float u_acc = 0.0f;
        for (int i = 0; i < D_MODEL; ++i) {
            g_acc += s_norm_x[i] * W_gate[h * D_MODEL + i];
            u_acc += s_norm_x[i] * W_up[h * D_MODEL + i];
        }
        // SwiGLU activation: SiLU(gate) * up
        float silu_gate = g_acc / (1.0f + expf(-g_acc));
        intermediate_ffn[t * FFN_HIDDEN + h] = silu_gate * u_acc;
    }
}

// Down projection and Final Output Residual accumulation
__global__ void ffn_down_residual_kernel(const float* __restrict__ intermediate_ffn,
                                         const float* __restrict__ W_down,
                                         const float* __restrict__ x_prime,
                                         float* __restrict__ output,
                                         int seq_len) {
    int t = blockIdx.x;
    int tid = threadIdx.x; // 0 to 511

    if (t >= seq_len) return;

    // Load full hidden vector representation sequentially into shared memory
    __shared__ float s_ffn[FFN_HIDDEN];
    for (int i = tid; i < FFN_HIDDEN; i += D_MODEL) {
        s_ffn[i] = intermediate_ffn[t * FFN_HIDDEN + i];
    }
    __syncthreads();

    float acc = 0.0f;
    for (int i = 0; i < FFN_HIDDEN; ++i) {
        acc += s_ffn[i] * W_down[tid * FFN_HIDDEN + i];
    }

    output[t * D_MODEL + tid] = x_prime[t * D_MODEL + tid] + acc;
}

// -----------------------------------------------------------------------------
// Main Solve Interface
// -----------------------------------------------------------------------------

extern "C" void solve(const float* x, float* output, const float* weights, const float* cos,
                      const float* sin, int seq_len) {
    
    // Allocate device scratch space buffers dynamically
    float *d_Q, *d_K, *d_V, *d_attn_out, *d_x_prime, *d_inter_ffn;
    cudaMalloc((void**)&d_Q, seq_len * D_MODEL * sizeof(float));
    cudaMalloc((void**)&d_K, seq_len * 128 * sizeof(float));
    cudaMalloc((void**)&d_V, seq_len * 128 * sizeof(float));
    cudaMalloc((void**)&d_attn_out, seq_len * D_MODEL * sizeof(float));
    cudaMalloc((void**)&d_x_prime, seq_len * D_MODEL * sizeof(float));
    cudaMalloc((void**)&d_inter_ffn, seq_len * FFN_HIDDEN * sizeof(float));


// Map weights pointers from the packed weights block buffer
const float* w1 = weights + OFF_W1;
const float* W_Q = weights + OFF_WQ;
const float* W_K = weights + OFF_WK;
const float* W_V = weights + OFF_WV;
const float* W_O = weights + OFF_WO;
const float* w2 = weights + OFF_W2;
const float* W_gate = weights + OFF_WGATE;
const float* W_up = weights + OFF_WUP;
const float* W_down = weights + OFF_WDOWN;
// 1. RMSNorm 1 + QKV Projections
rmsnorm_qkv_kernel<<<seq_len, D_MODEL>>>(x, w1, W_Q, W_K, W_V, d_Q, d_K, d_V, seq_len);
// 2. Rotary Position Embeddings (RoPE)
dim3 rope_blocks(seq_len, N_Q_HEADS); // 8 is max heads between Q and KV
apply_rope_kernel<<<rope_blocks, HEAD_DIM / 2>>>(d_Q, d_K, cos, sin, seq_len);
// 3. Grouped Query Attention (GQA) with Causal Mask
dim3 attn_blocks(seq_len, N_Q_HEADS);
size_t shared_scores_size = seq_len * sizeof(float);
attention_kernel<<<attn_blocks, HEAD_DIM, shared_scores_size>>>(d_Q, d_K, d_V, d_attn_out, seq_len);
// 4. Output Projection + First Residual Add
out_proj_residual_kernel<<<seq_len, D_MODEL>>>(d_attn_out, W_O, x, d_x_prime, seq_len);
// 5. RMSNorm 2 + SwiGLU FFN Layer 1 (Gate & Up Projections)
ffn_gate_up_kernel<<<seq_len, D_MODEL>>>(d_x_prime, w2, W_gate, W_up, d_inter_ffn, seq_len);
// 6. FFN Down Projection + Final Output Residual Add
ffn_down_residual_kernel<<<seq_len, D_MODEL>>>(d_inter_ffn, W_down, d_x_prime, output, seq_len);
// Cleanup resources
cudaFree(d_Q);
cudaFree(d_K);
cudaFree(d_V);
cudaFree(d_attn_out);
cudaFree(d_x_prime);
cudaFree(d_inter_ffn);
} 