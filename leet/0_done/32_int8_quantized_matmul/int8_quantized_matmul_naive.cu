#include <cuda_runtime.h>
#include <stdint.h>

// Naive baseline for challenge 32: the starter's dp4a GEMM (64x64 tile, 4x4
// per thread, exact zero-point epilogue) with ONE fix -- compute_row_sums_naive no
// longer does int32 reinterpret-cast loads at `A_row + k`, which are
// misaligned whenever K is not a multiple of 4 and fault the kernel from the
// first such case (poisoning the CUDA context).  The row sums are accumulated
// with scalar loads instead; everything else is the starter's shape.
// Measures 268.4 ms at the benchmark shape (256 G int8-MAC/s).

// Precomputes row sums of A: row_sums[i] = sum_{k=0}^{K-1} A[i, k]
__global__ void compute_row_sums_naive(const int8_t* __restrict__ A, int32_t* __restrict__ row_sums, int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;

    int32_t sum = 0;
    const int8_t* A_row = A + (size_t)row * K;

    // Scalar accumulation: A_row is not 4-byte aligned when K is odd, so the
    // starter's int32 reinterpret-cast loads are misaligned for K % 4 != 0.
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += static_cast<int32_t>(A_row[k]);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ int32_t shared_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) shared_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < (blockDim.x >> 5)) ? shared_sums[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) row_sums[row] = sum;
    }
}

// Precomputes column sums of B: col_sums[j] = sum_{k=0}^{K-1} B[k, j]
__global__ void compute_col_sums_naive(const int8_t* __restrict__ B, int32_t* __restrict__ col_sums, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
        sum += static_cast<int32_t>(B[(size_t)k * N + col]);
    }
    col_sums[col] = sum;
}

// Tiled INT8 GEMM with __dp4a and quantized epilogue
// Block tile: 64x64, Thread tile: 4x4, K-step: 16
template <int BM = 64, int BN = 64, int BK = 16, int TM = 4, int TN = 4>
__global__ void qgemm_dp4a_naive(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* __restrict__ C,
    const int32_t* __restrict__ row_sums,
    const int32_t* __restrict__ col_sums,
    int M, int N, int K,
    float combined_scale,
    int32_t z_A, int32_t z_B, int32_t z_C) {

    // Shared memory buffers:
    // As stores A[BM][BK] row-major (padded to avoid bank conflicts)
    // Bs stores B[BN][BK] transposed so K-chunks of 4 are contiguous for __dp4a
    __shared__ int8_t As[BM][BK + 4];
    __shared__ int8_t Bs[BN][BK + 4];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255 (16x16 threads)

    int ty = threadIdx.y * TM;
    int tx = threadIdx.x * TN;

    int32_t acc[TM][TN] = {0};

    // Loop over K in chunks of BK (16)
    for (int k_base = 0; k_base < K; k_base += BK) {
        // Load A tile (64 x 16 = 1024 elements -> 4 elements per thread)
        for (int i = tid; i < BM * BK; i += 256) {
            int r = i / BK;
            int c = i % BK;
            int global_r = by * BM + r;
            int global_c = k_base + c;
            As[r][c] = (global_r < M && global_c < K) ? A[(size_t)global_r * K + global_c] : 0;
        }

        // Load B tile (16 x 64 = 1024 elements -> transposed into Bs[col][k])
        for (int i = tid; i < BK * BN; i += 256) {
            int r = i / BN; // k offset within BK
            int c = i % BN; // n offset within BN
            int global_r = k_base + r;
            int global_c = bx * BN + c;
            Bs[c][r] = (global_r < K && global_c < N) ? B[(size_t)global_r * N + global_c] : 0;
        }

        __syncthreads();

        // Compute 4x4 register tile using __dp4a along BK in steps of 4
        #pragma unroll
        for (int k = 0; k < BK; k += 4) {
            int32_t a_frag[TM];
            int32_t b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = *reinterpret_cast<const int32_t*>(&As[ty + i][k]);
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = *reinterpret_cast<const int32_t*>(&Bs[tx + j][k]);
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __dp4a(a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Epilogue: apply zero-point offsets, scale, round-half-to-even, and clamp
    int32_t k_zA_zB = K * z_A * z_B;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_r = by * BM + ty + i;
        if (global_r >= M) continue;
        int32_t r_sum = row_sums[global_r];

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_c = bx * BN + tx + j;
            if (global_c >= N) continue;

            int32_t c_sum = col_sums[global_c];

            // Exact int32 dot product: sum((A - zA)(B - zB))
            int32_t exact_dot = acc[i][j] - z_B * r_sum - z_A * c_sum + k_zA_zB;

            // Scale in float32 and round-half-to-even using rintf
            float scaled = static_cast<float>(exact_dot) * combined_scale;
            int32_t rounded = static_cast<int32_t>(rintf(scaled)) + z_C;

            // Clamp to [-128, 127]
            rounded = max(-128, min(127, rounded));
            C[(size_t)global_r * N + global_c] = static_cast<int8_t>(rounded);
        }
    }
}

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {
    int32_t* d_row_sums = nullptr;
    int32_t* d_col_sums = nullptr;
    cudaMalloc(&d_row_sums, M * sizeof(int32_t));
    cudaMalloc(&d_col_sums, N * sizeof(int32_t));

    // 1. Precompute row sums of A and column sums of B
    compute_row_sums_naive<<<M, 256>>>(A, d_row_sums, M, K);
    compute_col_sums_naive<<<(N + 255) / 256, 256>>>(B, d_col_sums, K, N);

    // 2. Launch tiled __dp4a GEMM
    dim3 block(16, 16); // 256 threads per block, each computing a 4x4 tile -> 64x64 block output
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    float combined_scale = (scale_A * scale_B) / scale_C;

    qgemm_dp4a_naive<64, 64, 16, 4, 4><<<grid, block>>>(
        A, B, C, d_row_sums, d_col_sums, M, N, K,
        combined_scale, zero_point_A, zero_point_B, zero_point_C
    );

    cudaFree(d_row_sums);
    cudaFree(d_col_sums);
}
