#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Tile dimensions chosen for optimal shared memory occupancy and coalescing
#define TILE_SIZE 32

__global__ void batched_gemm_fp16_kernel(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    // Determine which matrix in the batch this block handles
    int batch_idx = blockIdx.z;
    
    // Shift base pointers for the current batch element
    const half* A_batch = A + batch_idx * M * K;
    const half* B_batch = B + batch_idx * K * N;
    half* C_batch = C + batch_idx * M * N;

    // Allocate shared memory for tiling
    __shared__ half s_A[TILE_SIZE][TILE_SIZE];
    __shared__ half s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Use FP32 for accumulation to ensure numerical precision
    float accum = 0.0f;

    // Loop over tiles of the K-dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load element into shared memory for A
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            s_A[threadIdx.y][threadIdx.x] = A_batch[row * K + (t * TILE_SIZE + threadIdx.x)];
        } else {
            s_A[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }

        // Load element into shared memory for B
        if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
            s_B[threadIdx.y][threadIdx.x] = B_batch[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }

        // Synchronize to make sure the tiles are loaded completely
        __syncthreads();

        // Compute the partial dot product using FP32 accumulation
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            accum += __half2float(s_A[threadIdx.y][k]) * __half2float(s_B[k][threadIdx.x]);
        }

        // Synchronize to ensure computation is done before loading the next tile
        __syncthreads();
    }

    // Write out the final FP32 accumulated result back into FP16 array C
    if (row < M && col < N) {
        C_batch[row * N + col] = __float2half(accum);
    }
}

extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    // Configure thread block dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);
    
    // Configure grid dimensions, embedding the batch size along the Z axis
    dim3 numBlocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        BATCH
    );

    // Launch kernel
    batched_gemm_fp16_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K);
}
