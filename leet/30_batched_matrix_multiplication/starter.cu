#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void batched_matmul_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int BATCH, int M, int N, int K) {
    // Identify the batch slice from blockIdx.z
    int b = blockIdx.z;
    if (b >= BATCH) return;

    // Shared memory tiles for sub-matrices of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate row and column indices for the output element in C[b]
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Pointers to the start of the current batch's matrices
    const float* A_b = A + (size_t)b * M * K;
    const float* B_b = B + (size_t)b * K * N;
    float* C_b       = C + (size_t)b * M * N;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over all tiles along the K dimension
    for (int t = 0; t < num_tiles; ++t) {
        int tiled_col_A = t * TILE_SIZE + threadIdx.x;
        int tiled_row_B = t * TILE_SIZE + threadIdx.y;

        // Load tile from A into shared memory with boundary check
        if (row < M && tiled_col_A < K) {
            As[threadIdx.y][threadIdx.x] = A_b[row * K + tiled_col_A];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory with boundary check
        if (tiled_row_B < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B_b[tiled_row_B * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure both tiles are completely loaded
        __syncthreads();

        // Accumulate partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize before loading the next set of tiles
        __syncthreads();
    }

    // Write the final result to global memory
    if (row < M && col < N) {
        C_b[row * N + col] = sum;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE,
                 BATCH);

    batched_matmul_kernel<<<gridDim, blockDim>>>(A, B, C, BATCH, M, N, K);
}
