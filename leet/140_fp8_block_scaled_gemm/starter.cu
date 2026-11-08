#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Fast OCP E4M3 (1 sign, 4 exp, 3 mantissa, bias=7, finite) to float32
__device__ __forceinline__ float decode_e4m3(uint8_t val) {
    uint32_t s = (val >> 7) & 0x1u;
    uint32_t e = (val >> 3) & 0xFu;
    uint32_t m = val & 0x7u;

    if (e == 0) {
        // Subnormal: (-1)^s * 2^(-6) * (m / 8) = (-1)^s * m * 2^(-9)
        float res = static_cast<float>(m) * (1.0f / 512.0f);
        return s ? -res : res;
    }
    // Normal: re-bias exponent from 7 to 127 (+120) and shift mantissa from 3 to 23 (+20)
    uint32_t f_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    return __uint_as_float(f_bits);
}

// Tile dimensions: 128x128 output tile per block, 64 elements along K per shared memory stage
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void fp8_block_scaled_gemm_kernel(
    const uint8_t* __restrict__ x_q,
    const float* __restrict__ x_scales,
    const uint8_t* __restrict__ w_q,
    const float* __restrict__ w_scales,
    __half* __restrict__ y,
    int M, int N, int K) {

    // Pad inner dimension by 4 floats to avoid shared memory bank conflicts
    __shared__ float As[BM][BK + 4];
    __shared__ float Bs[BN][BK + 4];

    const int bx = blockIdx.x; // Tile index along N
    const int by = blockIdx.y; // Tile index along M

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255 (16x16 threads)
    const int ty = threadIdx.y; // 0..15
    const int tx = threadIdx.x; // 0..15

    const int row_base = by * BM;
    const int col_base = bx * BN;
    const int num_scale_blocks = K >> 7; // K / 128

    // Total accumulators for the 8x8 sub-tile owned by this thread
    float total_acc[TM][TN] = {{0.0f}};
    // Block-local accumulators (reset every 128 elements along K)
    float block_acc[TM][TN] = {{0.0f}};

    // Each thread loads 32 elements of A (128 * 64 / 256) and 32 elements of B per K-tile
    // Organized as 8 rows x 4 contiguous bytes per thread
    const int load_row_start = (tid >> 4) * 8; // (tid / 16) * 8 -> 0, 8, ..., 120
    const int load_col_start = (tid & 15) * 4; // (tid % 16) * 4 -> 0, 4, ..., 60

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // 1. Cooperative global-to-shared load & E4M3 decode for A (x_q) and B (w_q)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r_a = row_base + load_row_start + i;
            int c = k_tile + load_col_start;
            if (r_a < M) {
                uint32_t packed = *reinterpret_cast<const uint32_t*>(&x_q[r_a * K + c]);
                As[load_row_start + i][load_col_start + 0] = decode_e4m3(packed & 0xFFu);
                As[load_row_start + i][load_col_start + 1] = decode_e4m3((packed >> 8) & 0xFFu);
                As[load_row_start + i][load_col_start + 2] = decode_e4m3((packed >> 16) & 0xFFu);
                As[load_row_start + i][load_col_start + 3] = decode_e4m3((packed >> 24) & 0xFFu);
            } else {
                As[load_row_start + i][load_col_start + 0] = 0.0f;
                As[load_row_start + i][load_col_start + 1] = 0.0f;
                As[load_row_start + i][load_col_start + 2] = 0.0f;
                As[load_row_start + i][load_col_start + 3] = 0.0f;
            }

            int r_b = col_base + load_row_start + i;
            if (r_b < N) {
                uint32_t packed = *reinterpret_cast<const uint32_t*>(&w_q[r_b * K + c]);
                Bs[load_row_start + i][load_col_start + 0] = decode_e4m3(packed & 0xFFu);
                Bs[load_row_start + i][load_col_start + 1] = decode_e4m3((packed >> 8) & 0xFFu);
                Bs[load_row_start + i][load_col_start + 2] = decode_e4m3((packed >> 16) & 0xFFu);
                Bs[load_row_start + i][load_col_start + 3] = decode_e4m3((packed >> 24) & 0xFFu);
            } else {
                Bs[load_row_start + i][load_col_start + 0] = 0.0f;
                Bs[load_row_start + i][load_col_start + 1] = 0.0f;
                Bs[load_row_start + i][load_col_start + 2] = 0.0f;
                Bs[load_row_start + i][load_col_start + 3] = 0.0f;
            }
        }

        __syncthreads();

        // 2. Register-blocked outer product across BK = 64
        float reg_a[TM];
        float reg_b[TN];

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                reg_a[m] = As[ty * TM + m][k];
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                reg_b[n] = Bs[tx * TN + n][k];
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    block_acc[m][n] = fmaf(reg_a[m], reg_b[n], block_acc[m][n]);
                }
            }
        }

        __syncthreads();

        // 3. Every 128 elements (every 2nd BK=64 tile), apply block scales and fold into total_acc
        if (((k_tile + BK) & 127) == 0) {
            int sb = k_tile >> 7; // Current 128-element scale block index

            float sx[TM];
            float sw[TN];

            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                int gm = row_base + ty * TM + m;
                sx[m] = (gm < M) ? x_scales[gm * num_scale_blocks + sb] : 0.0f;
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                int gn = col_base + tx * TN + n;
                sw[n] = (gn < N) ? w_scales[gn * num_scale_blocks + sb] : 0.0f;
            }

            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    total_acc[m][n] = fmaf(block_acc[m][n], sx[m] * sw[n], total_acc[m][n]);
                    block_acc[m][n] = 0.0f;
                }
            }
        }
    }

    // 4. Write final float16 results to global memory
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int gm = row_base + ty * TM + m;
        if (gm < M) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                int gn = col_base + tx * TN + n;
                if (gn < N) {
                    y[gm * N + gn] = __float2half(total_acc[m][n]);
                }
            }
        }
    }
}

// x_q, x_scales, w_q, w_scales, y are device pointers
extern "C" void solve(const uint8_t* x_q, const float* x_scales, const uint8_t* w_q,
                      const float* w_scales, __half* y, int M, int N, int K) {
    dim3 block(16, 16); // 256 threads per block
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    fp8_block_scaled_gemm_kernel<<<grid, block>>>(x_q, x_scales, w_q, w_scales, y, M, N, K);
}
