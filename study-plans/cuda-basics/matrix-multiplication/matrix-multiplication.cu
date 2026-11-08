#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // Allocate shared memory for the tiles
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    // Local thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global matrix coordinates for the output element
    int Row = blockIdx.y * TILE_SIZE + ty;
    int Col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;

    // Loop over the K-dimension in TILE_SIZE chunks
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        
        // 1. Cooperative Load: Each thread loads one element of A and B into shared memory.
        // Bounds checking is required because M, N, or K might not be multiples of TILE_SIZE.
        int a_col = t * TILE_SIZE + tx;
        if (Row < M && a_col < K) {
            s_A[ty][tx] = A[Row * K + a_col];
        } else {
            s_A[ty][tx] = 0.0f; // Pad with zeros to handle boundary tiles safely
        }

        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && Col < N) {
            s_B[ty][tx] = B[b_row * N + Col];
        } else {
            s_B[ty][tx] = 0.0f; // Pad with zeros
        }

        // 2. Synchronize to ensure the entire block has finished loading the tiles
        __syncthreads();

        // 3. Compute Phase: Multiply the shared memory tiles
        // We accumulate the partial dot product for this tile step.
        // #pragma unroll is highly recommended here to reduce loop overhead.
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }

        // 4. Synchronize again before the next iteration
        // This prevents threads that finish the compute phase early from 
        // overwriting s_A and s_B while other threads are still computing.
        __syncthreads();
    }

    // Write the final accumulated value to global memory
    if (Row < M && Col < N) {
        C[Row * N + Col] = acc;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel_tiled<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}