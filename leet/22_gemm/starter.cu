#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void gemm_fp16_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 half* __restrict__ C,
                                 int M, int N, int K,
                                 float alpha, float beta) {
    // Shared memory tiles for A and B stored as float to prevent repeated conversions
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // FP32 accumulator for higher precision
    float acc = 0.0f;

    // Loop over all tiles required to compute the C element
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        // Collaborative loading of A tile into shared memory with bounds check
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = __half2float(A[row * K + a_col]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Collaborative loading of B tile into shared memory with bounds check
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __half2float(B[b_row * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure the entire tile is loaded before computation
        __syncthreads();

        // Accumulate partial dot product for the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
        }

        // Synchronize before loading the next tile in the next iteration
        __syncthreads();
    }

    // Write the scaled result back to global memory in FP16
    if (row < M && col < N) {
        float c_val = __half2float(C[row * N + col]);
        float res = alpha * acc + beta * c_val;
        C[row * N + col] = __float2half(res);
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_fp16_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}
