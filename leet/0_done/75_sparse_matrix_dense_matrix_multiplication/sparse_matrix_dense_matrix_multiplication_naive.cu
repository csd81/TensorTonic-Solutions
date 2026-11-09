#include <cuda_runtime.h>

// Naive baseline for challenge 75, kept as the starter's shape: a 32x32
// shared tile with one output per thread, i.e. two shared-memory loads per
// multiply-add.  It ignores sparsity entirely.  Measures 28.0 ms at
// M=4096, N=2048, K=512 (0.307 TFLOP/s dense-equivalent).

#define TILE_SIZE 32

__global__ void tiled_spmm_dense_naive(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of C to work on
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over the tiles of A and B required to compute C[row, col]
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        // Collaborative loading of A tile into shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < N) {
            As[ty][tx] = A[row * N + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Collaborative loading of B tile into shared memory
        int b_row = t * TILE_SIZE + ty;
        if (b_row < N && col < K) {
            Bs[ty][tx] = B[b_row * K + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_val = As[ty][k];
            // Optional branchless FMA; skipping zeros here often causes warp divergence
            sum += a_val * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write final result to global memory
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz) {
    // Unused parameter warning suppression
    (void)nnz;

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    tiled_spmm_dense_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
