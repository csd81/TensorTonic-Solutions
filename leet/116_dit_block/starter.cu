#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define D_MODEL 512
#define N_HEADS 8
#define HEAD_DIM 64
#define MLP_HIDDEN 2048
#define TILE_SIZE 32

// Weight offsets
#define OFF_W_ADA  0
#define OFF_B_ADA  1572864
#define OFF_W_QKV  1575936
#define OFF_B_QKV  2362368
#define OFF_W_O    2363904
#define OFF_B_O    2626048
#define OFF_W_FC1  2626560
#define OFF_B_FC1  3675136
#define OFF_W_FC2  3677184
#define OFF_B_FC2  4725760

__device__ __forceinline__ float silu_fn(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu_tanh_fn(float x) {
    const float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2 / pi)
    const float kCoeff = 0.044715f;
    float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// 1. Compute adaLN modulation parameters: [B, 512] -> SiLU -> Linear(512 -> 3072) -> [B, 3072]
__global__ void adaln_mod_kernel(const float* __restrict__ c,
                                 const float* __restrict__ w_ada,
                                 const float* __restrict__ b_ada,
                                 float* __restrict__ mod_out,
                                 int batch_size) {
    int b = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    __shared__ float s_c[D_MODEL];
    for (int i = threadIdx.x; i < D_MODEL; i += blockDim.x) {
        s_c[i] = silu_fn(c[b * D_MODEL + i]);
    }
    __syncthreads();

    if (col < 3072) {
        const float* w_row = w_ada + col * D_MODEL;
        float sum = b_ada[col];
        #pragma unroll 4
        for (int k = 0; k < D_MODEL; ++k) {
            sum += s_c[k] * w_row[k];
        }
        mod_out[b * 3072 + col] = sum;
    }
}

// 2. Fused LayerNorm (no affine) + adaLN Modulation: h = LN(x) * (1 + alpha) + gamma
// mod_offset: 0 for MSA (gamma at +0, alpha at +512), 1536 for MLP (gamma at +1536, alpha at +2048)
__global__ void layernorm_modulate_kernel(const float* __restrict__ x,
                                          const float* __restrict__ mod,
                                          float* __restrict__ h,
                                          int seq_len,
                                          int mod_offset) {
    int token_idx = blockIdx.x; // 0 .. B * T - 1
    int b = token_idx / seq_len;
    int tid = threadIdx.x; // 256 threads, 2 elements per thread

    const float* x_row = x + token_idx * D_MODEL;
    float* h_row = h + token_idx * D_MODEL;

    float v0 = x_row[tid];
    float v1 = x_row[tid + 256];

    __shared__ float s_reduce[256];
    s_reduce[tid] = v0 + v1;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    float mean = s_reduce[0] / (float)D_MODEL;
    __syncthreads();

    float d0 = v0 - mean;
    float d1 = v1 - mean;
    s_reduce[tid] = d0 * d0 + d1 * d1;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    float inv_std = rsqrtf(s_reduce[0] / (float)D_MODEL + 1e-6f);

    const float* gamma = mod + b * 3072 + mod_offset;
    const float* alpha = mod + b * 3072 + mod_offset + D_MODEL;

    h_row[tid]       = (d0 * inv_std) * (1.0f + alpha[tid])       + gamma[tid];
    h_row[tid + 256] = (d1 * inv_std) * (1.0f + alpha[tid + 256]) + gamma[tid + 256];
}

// 3. Tiled Linear Projection: Y[M, N] = Act(X[M, K] * W[N, K]^T + B[N])
// Optionally fuses gated residual: Y[M, N] = residual[M, N] + gate[b, N] * (X * W^T + B)
template <bool APPLY_GELU, bool FUSE_GATED_RESIDUAL>
__global__ void linear_kernel(const float* __restrict__ X,
                              const float* __restrict__ W,
                              const float* __restrict__ B,
                              const float* __restrict__ residual,
                              const float* __restrict__ mod,
                              float* __restrict__ Y,
                              int M, int N, int K,
                              int seq_len, int gate_offset) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE + 1]; // +1 padding avoids bank conflicts

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_k_a = t * TILE_SIZE + threadIdx.x;
        if (row < M && tiled_k_a < K) {
            sA[threadIdx.y][threadIdx.x] = X[row * K + tiled_k_a];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load W[col_w, k_w] into sB[threadIdx.y][threadIdx.x]
        int w_row = blockIdx.x * TILE_SIZE + threadIdx.y;
        int w_col = t * TILE_SIZE + threadIdx.x;
        if (w_row < N && w_col < K) {
            sB[threadIdx.y][threadIdx.x] = W[w_row * K + w_col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sA[threadIdx.y][k] * sB[threadIdx.x][k];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = acc + B[col];
        if (APPLY_GELU) {
            val = gelu_tanh_fn(val);
        }
        if (FUSE_GATED_RESIDUAL) {
            int b = row / seq_len;
            float g = mod[b * 3072 + gate_offset + col];
            val = residual[row * N + col] + g * val;
        }
        Y[row * N + col] = val;
    }
}

// 4. Multi-Head Self-Attention (Bidirectional, 8 heads, head_dim = 64)
// Grid: (seq_len, N_HEADS, batch_size), Block: 64 threads (1 per head_dim element)
__global__ void mha_kernel(const float* __restrict__ qkv,
                           float* __restrict__ attn_out,
                           int seq_len) {
    int t_q = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int d = threadIdx.x; // 0 .. 63

    if (t_q >= seq_len) return;

    const int qkv_stride = 3 * D_MODEL; // 1536
    const float scale = 0.125f;         // 1 / sqrt(64)

    // Load Q[b, t_q, h, d]
    float q_val = qkv[(b * seq_len + t_q) * qkv_stride + h * HEAD_DIM + d];

    extern __shared__ float smem[];
    float* s_scores = smem;               // size: seq_len
    float* s_red = smem + seq_len;        // size: 64

    float max_score = -FLT_MAX;

    // Pass 1: Compute attention scores Q * K^T / sqrt(64) and find row max
    for (int t_k = 0; t_k < seq_len; ++t_k) {
        float k_val = qkv[(b * seq_len + t_k) * qkv_stride + D_MODEL + h * HEAD_DIM + d];
        s_red[d] = q_val * k_val;
        __syncthreads();

        for (int stride = 32; stride > 0; stride >>= 1) {
            if (d < stride) s_red[d] += s_red[d + stride];
            __syncthreads();
        }

        if (d == 0) {
            float score = s_red[0] * scale;
            s_scores[t_k] = score;
            if (score > max_score) max_score = score;
        }
        __syncthreads();
    }

    // Broadcast max_score to all threads in block
    if (d == 0) s_red[0] = max_score;
    __syncthreads();
    max_score = s_red[0];

    // Pass 2: Exponentiate and compute softmax denominator
    float local_sum = 0.0f;
    for (int t_k = d; t_k < seq_len; t_k += HEAD_DIM) {
        float e = expf(s_scores[t_k] - max_score);
        s_scores[t_k] = e;
        local_sum += e;
    }
    s_red[d] = local_sum;
    __syncthreads();

    for (int stride = 32; stride > 0; stride >>= 1) {
        if (d < stride) s_red[d] += s_red[d + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / s_red[0];

    // Pass 3: Weighted sum over V
    float out_val = 0.0f;
    for (int t_k = 0; t_k < seq_len; ++t_k) {
        float prob = s_scores[t_k] * inv_sum;
        float v_val = qkv[(b * seq_len + t_k) * qkv_stride + 2 * D_MODEL + h * HEAD_DIM + d];
        out_val += prob * v_val;
    }

    attn_out[(b * seq_len + t_q) * D_MODEL + h * HEAD_DIM + d] = out_val;
}

// x, c, output, weights are device pointers
extern "C" void solve(const float* x, const float* c, float* output, const float* weights,
                      int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;

    // Workspace allocation
    float *d_mod, *d_norm, *d_qkv, *d_attn, *d_x_prime, *d_mlp_hid;
    size_t mod_bytes     = batch_size * 3072 * sizeof(float);
    size_t tok_512_bytes = total_tokens * D_MODEL * sizeof(float);
    size_t qkv_bytes     = total_tokens * 1536 * sizeof(float);
    size_t mlp_bytes     = total_tokens * MLP_HIDDEN * sizeof(float);

    cudaMalloc(&d_mod, mod_bytes);
    cudaMalloc(&d_norm, tok_512_bytes);
    cudaMalloc(&d_qkv, qkv_bytes);
    cudaMalloc(&d_attn, tok_512_bytes);
    cudaMalloc(&d_x_prime, tok_512_bytes);
    cudaMalloc(&d_mlp_hid, mlp_bytes);

    const float* w_ada = weights + OFF_W_ADA;
    const float* b_ada = weights + OFF_B_ADA;
    const float* w_qkv = weights + OFF_W_QKV;
    const float* b_qkv = weights + OFF_B_QKV;
    const float* w_o   = weights + OFF_W_O;
    const float* b_o   = weights + OFF_B_O;
    const float* w_fc1 = weights + OFF_W_FC1;
    const float* b_fc1 = weights + OFF_B_FC1;
    const float* w_fc2 = weights + OFF_W_FC2;
    const float* b_fc2 = weights + OFF_B_FC2;

    dim3 block_tile(TILE_SIZE, TILE_SIZE);

    // Step 1: Compute 6 adaLN-Zero modulation vectors [B, 3072]
    dim3 grid_mod((3072 + 255) / 256, batch_size);
    adaln_mod_kernel<<<grid_mod, 256>>>(c, w_ada, b_ada, d_mod, batch_size);

    // Step 2: h = LN(x) * (1 + alpha_msa) + gamma_msa
    layernorm_modulate_kernel<<<total_tokens, 256>>>(x, d_mod, d_norm, seq_len, 0);

    // Step 3: [Q | K | V] = h * W_qkv^T + b_qkv
