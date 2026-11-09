#include <cuda_runtime.h>

// Tiling dimensions optimized for modern GPU architectures
#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

__global__ void grouped_gemm_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const int* __restrict__ group_offsets, 
    float* __restrict__ C, 
    int G, int M_total, int K, int N) 
{
    // Global row index in matrix C and A
    int global_row = blockIdx.y * BLOCK_M + threadIdx.y;
    // Global column index in matrix C and B
    int global_col = blockIdx.x * BLOCK_N + threadIdx.x;

    if (global_row >= M_total || global_col >= N) {
        return;
    }

    // Binary search to find which group this global_row belongs to.
    // group_offsets is sorted and non-decreasing.
    int low = 0;
    int high = G - 1;
    int g = 0;
    
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (group_offsets[mid] <= global_row) {
            g = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    // Check if the current row falls within the current group boundaries
    int s = group_offsets[g];
    int e = group_offsets[g + 1];
    if (global_row >= e) {
        return; // Falls into an empty/invalid slot
    }

    // Accumulator for C[global_row, global_col]
    float sum = 0.0f;

    // Pointer to the specific weight matrix for this group
    // B has shape [G, K, N], so offset is g * K * N
    const float* B_group = B + (size_t)g * K * N;

    // Shared memory for cooperative caching of sub-tiles
    __shared__ float s_A[BLOCK_M][BLOCK_K];
    __shared__ float s_B[BLOCK_K][BLOCK_N];

    // Loop over the inner dimension K in tile steps
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        
        // 1. Cooperative load from Matrix A into shared memory
        int col_A = k_tile + threadIdx.x;
        if (col_A < K) {
            s_A[threadIdx.y][threadIdx.x] = A[(size_t)global_row * K + col_A];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 2. Cooperative load from Matrix B_group into shared memory
        int row_B = k_tile + threadIdx.y;
        if (row_B < K) {
            s_B[threadIdx.y][threadIdx.x] = B_group[(size_t)row_B * N + global_col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize threads to make sure tiles are fully loaded
        __syncthreads();

        // 3. Compute the dot product for this tile step
        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // Synchronize threads before loading the next tile step
        __syncthreads();
    }

    // Write out the computed element to global memory matrix C
    C[(size_t)global_row * N + global_col] = sum;
}

extern "C" void solve(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) 
{
    // Define block execution geometry
    dim3 threadsPerBlock(BLOCK_N, BLOCK_M);
    
    // Calculate required grid dimensions to fully cover output dimensions
    dim3 numBlocks(
        (N + BLOCK_N - 1) / BLOCK_N,
        (M_total + BLOCK_M - 1) / BLOCK_M
    );

    // Launch execution kernel on default stream
    grouped_gemm_kernel<<<numBlocks, threadsPerBlock>>>(A, B, group_offsets, C, G, M_total, K, N);
}
