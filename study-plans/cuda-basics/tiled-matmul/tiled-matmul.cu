#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Allocate shared memory for the tiles
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    // Calculate global row and column indices for the output matrix C
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    
    // Number of tile phases needed to cover the K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        // 1. Cooperative load into shared memory tile for A
        int a_col = t * TILE_DIM + threadIdx.x;
        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 2. Cooperative load into shared memory tile for B
        int b_row = t * TILE_DIM + threadIdx.y;
        if (b_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads have finished loading the current tile
        __syncthreads();

        // 3. Accumulate partial dot products from the loaded shared memory tiles
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // Synchronize to ensure all threads are done computing before overwriting the tiles in the next phase
        __syncthreads();
    }

    // Write the final accumulated value to global memory with boundary checks
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    tiled_matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}