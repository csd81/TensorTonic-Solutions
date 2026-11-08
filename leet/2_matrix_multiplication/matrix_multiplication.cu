#include <cuda_runtime.h>

// Register-blocked tiled matrix multiplication.
//   Each block has 16x16 = 256 threads and computes a 64x64 output tile.
//   Each thread computes TM x TN = 4 x 4 outputs.
//   A and B tiles of size 64xBK and BKx64 are staged in shared memory.
//
// C = A * B, row-major: A is M x N, B is N x K, C is M x K.

#define BM 64  // output tile rows (M dimension)
#define BN 64  // output tile columns (K dimension)
#define BK 16  // reduction tile depth (N dimension)
#define TM 4   // outputs per thread along M
#define TN 4   // outputs per thread along K

__global__ void __launch_bounds__(256) matrix_multiplication_kernel(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;

    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < N; k0 += BK) {
        // Load A tile: BM x BK values (64 * 16 = 1024, 4 per thread).
        for (int idx = tid; idx < BM * BK; idx += nthreads) {
            const int r = idx / BK;
            const int c = idx - r * BK;
            const int gr = blockIdx.y * BM + r;
            const int gc = k0 + c;
            s_A[r][c] = (gr < M && gc < N)
                            ? A[static_cast<size_t>(gr) * N + gc]
                            : 0.0f;
        }

        // Load B tile: BK x BN values (16 * 64 = 1024, 4 per thread).
        for (int idx = tid; idx < BK * BN; idx += nthreads) {
            const int r = idx / BN;
            const int c = idx - r * BN;
            const int gr = k0 + r;
            const int gc = blockIdx.x * BN + c;
            s_B[r][c] = (gr < N && gc < K)
                            ? B[static_cast<size_t>(gr) * K + gc]
                            : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM];
            float b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = s_A[ty * TM + i][kk];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = s_B[kk][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = row0 + i;
        if (gr >= M) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = col0 + j;
            if (gc < K) {
                C[static_cast<size_t>(gr) * K + gc] = acc[i][j];
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + BN - 1) / BN, (M + BM - 1) / BM);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
