#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Tile sizes tuned for 32-bit float shared memory GEMMs
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;

// Kernel 1: Count tokens per expert
__global__ void count_experts_kernel(const int* __restrict__ topk_ids, int* __restrict__ counts, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        atomicAdd(&counts[topk_ids[idx]], 1);
    }
}

// Kernel 2: Scatter token indices and weights into expert-grouped order
__global__ void group_tokens_kernel(
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int* __restrict__ expert_cursor,
    int* __restrict__ sorted_token_ids,
    float* __restrict__ sorted_weights,
    int M, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        int e = topk_ids[idx];
        int pos = atomicAdd(&expert_cursor[e], 1);
        sorted_token_ids[pos] = idx / K;
        sorted_weights[pos] = topk_weights[idx];
    }
}

// Kernel 3: Fused Gather + Gate/Up GEMM + SwiGLU Activation
// Grid: (ceil(I / BN), total_m_tiles, 1)
// Block: (BN / TN, BM / TM, 1) -> (16, 16, 1) = 256 threads
__global__ void fused_gate_up_swiglu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    float* __restrict__ inter,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ tile_expert_id,
    const int* __restrict__ tile_m_start,
    const int* __restrict__ expert_offsets,
    int H, int I
) {
    int tile_idx = blockIdx.y;
    int e = tile_expert_id[tile_idx];
    int m_start = tile_m_start[tile_idx];
    int m_end = expert_offsets[e + 1];
    int n_start = blockIdx.x * BN;

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15
    int tid = ty * blockDim.x + tx;

    __shared__ float As[BM][BK + 1];
    __shared__ float B_gate[BK][BN + 1];
    __shared__ float B_up[BK][BN + 1];

    float acc_gate[TM][TN] = {0.0f};
    float acc_up[TM][TN] = {0.0f};

    const float* wg_e = w_gate + (size_t)e * H * I;
    const float* wu_e = w_up + (size_t)e * H * I;

    for (int k_base = 0; k_base < H; k_base += BK) {
        // Load As (BM x BK) -> 64 * 16 = 1024 floats (4 per thread)
        for (int i = 0; i < 4; ++i) {
            int linear_idx = tid * 4 + i;
            int r = linear_idx / BK;
            int c = linear_idx % BK;
            int global_m = m_start + r;
            int global_k = k_base + c;
            if (global_m < m_end && global_k < H) {
                int token_id = sorted_token_ids[global_m];
                As[r][c] = x[(size_t)token_id * H + global_k];
            } else {
                As[r][c] = 0.0f;
            }
        }

        // Load B_gate & B_up (BK x BN) -> 16 * 64 = 1024 floats (4 per thread)
        for (int i = 0; i < 4; ++i) {
            int linear_idx = tid * 4 + i;
            int r = linear_idx / BN;
            int c = linear_idx % BN;
            int global_k = k_base + r;
            int global_n = n_start + c;
            if (global_k < H && global_n < I) {
                size_t w_idx = (size_t)global_k * I + global_n;
                B_gate[r][c] = wg_e[w_idx];
                B_up[r][c] = wu_e[w_idx];
            } else {
                B_gate[r][c] = 0.0f;
                B_up[r][c] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM];
            float bg_reg[TN];
            float bu_reg[TN];

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                a_reg[tm] = As[ty * TM + tm][k];
            }
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                bg_reg[tn] = B_gate[k][tx * TN + tn];
                bu_reg[tn] = B_up[k][tx * TN + tn];
            }
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    acc_gate[tm][tn] = fmaf(a_reg[tm], bg_reg[tn], acc_gate[tm][tn]);
                    acc_up[tm][tn] = fmaf(a_reg[tm], bu_reg[tn], acc_up[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // Apply SwiGLU and store to intermediate buffer
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int global_m = m_start + ty * TM + tm;
        if (global_m < m_end) {
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                int global_n = n_start + tx * TN + tn;
                if (global_n < I) {
                    float g = acc_gate[tm][tn];
                    float u = acc_up[tm][tn];
                    float silu = g / (1.0f + expf(-g));
                    inter[(size_t)global_m * I + global_n] = silu * u;
                }
            }
        }
    }
}

// Kernel 4: Fused Down GEMM + Weighted Scatter-Add
__global__ void fused_down_scatter_kernel(
    const float* __restrict__ inter,
    const float* __restrict__ w_down,
    float* __restrict__ output,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    const int* __restrict__ tile_expert_id,
    const int* __restrict__ tile_m_start,
    const int* __restrict__ expert_offsets,
    int H, int I
) {
    int tile_idx = blockIdx.y;
    int e = tile_expert_id[tile_idx];
    int m_start = tile_m_start[tile_idx];
    int m_end = expert_offsets[e + 1];
    int n_start = blockIdx.x * BN;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    float acc[TM][TN] = {0.0f};
    const float* wd_e = w_down + (size_t)e * I * H;

    for (int k_base = 0; k_base < I; k_base += BK) {
        // Load As from inter (BM x BK)
        for (int i = 0; i < 4; ++i) {
            int linear_idx = tid * 4 + i;
            int r = linear_idx / BK;
            int c = linear_idx % BK;
            int global_m = m_start + r;
            int global_k = k_base + c;
            if (global_m < m_end && global_k < I) {
                As[r][c] = inter[(size_t)global_m * I + global_k];
            } else {
                As[r][c] = 0.0f;
            }
        }

        // Load Bs from w_down[e] (BK x BN)
        for (int i = 0; i < 4; ++i) {
            int linear_idx = tid * 4 + i;
            int r = linear_idx / BN;
            int c = linear_idx % BN;
            int global_k = k_base + r;
            int global_n = n_start + c;
            if (global_k < I && global_n < H) {
                Bs[r][c] = wd_e[(size_t)global_k * H + global_n];
            } else {
                Bs[r][c] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM];
            float b_reg[TN];

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                a_reg[tm] = As[ty * TM + tm][k];
            }
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                b_reg[tn] = Bs[k][tx * TN + tn];
            }
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    acc[tm][tn] = fmaf(a_reg[tm], b_reg[tn], acc[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // Weight and scatter-add into output
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int global_m = m_start + ty * TM + tm;
        if (global_m < m_end) {
            int token_id = sorted_token_ids[global_m];
            float weight = sorted_weights[global_m];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                int global_n = n_start + tx * TN + tn;
                if (global_n < H) {
                    float val = acc[tm][tn] * weight;
                    atomicAdd(&output[(size_t)token_id * H + global_n], val);
                }
            }
        }
    }
}

extern "C" void solve(const float* x, const int* topk_ids, const float* topk_weights,
                      const float* w_gate, const float* w_up, const float* w_down, float* output,
                      int T, int H, int I, int E, int K) {
    int M = T * K;
    if (M == 0) {
        cudaMemset(output, 0, (size_t)T * H * sizeof(float));
        return;
    }

    // 1. Count expert assignments on GPU
    int* d_counts;
    cudaMalloc(&d_counts, E * sizeof(int));
    cudaMemset(d_counts, 0, E * sizeof(int));

    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    count_experts_kernel<<<blocks, threads>>>(topk_ids, d_counts, M);

    // 2. Compute prefix sums & tile descriptors (E <= 64, trivial host overhead)
    std::vector<int> h_counts(E);
    cudaMemcpy(h_counts.data(), d_counts, E * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> h_offsets(E + 1, 0);
    std::vector<int> h_tile_expert_id;
    std::vector<int> h_tile_m_start;

    for (int e = 0; e < E; ++e) {
        h_offsets[e + 1] = h_offsets[e] + h_counts[e];
        for (int m = h_offsets[e]; m < h_offsets[e + 1]; m += BM) {
            h_tile_expert_id.push_back(e);
            h_tile_m_start.push_back(m);
        }
    }

    int num_m_tiles = static_cast<int>(h_tile_expert_id.size());

    // Allocate workspace buffers
    int *d_offsets, *d_cursor, *d_sorted_token_ids, *d_tile_expert_id, *d_tile_m_start;
    float *d_sorted_weights, *d_inter;

    cudaMalloc(&d_offsets, (E + 1) * sizeof(int));
    cudaMalloc(&d_cursor, E * sizeof(int));
    cudaMalloc(&d_sorted_token_ids, M * sizeof(int));
    cudaMalloc(&d_sorted_weights, M * sizeof(float));
    cudaMalloc(&d_tile_expert_id, num_m_tiles * sizeof(int));
