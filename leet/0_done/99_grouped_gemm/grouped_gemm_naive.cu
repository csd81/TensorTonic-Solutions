#include <cuda_runtime.h>

// Naive baseline for challenge 99: the starter's 16x16-tile, one-output-per-
// thread kernel with the shared-B race fixed.  The starter loads B into a
// per-block shared tile from each thread's own group's weight matrix, so a
// block whose rows span a group boundary computes every row with whichever
// group's B values were written last (verified: with groups {0,2,4} the
// second group's rows get the first group's product).  Here each output
// reads B directly from its own group's matrix, which is correct for any
// group layout but keeps the starter's plain structure and speed class.

// Tiling dimensions optimized for modern GPU architectures
#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

__global__ void grouped_gemm_naive_kernel(
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

    // Loop over the inner dimension K; the A tile is shared, B is read
    // directly from this row's own group matrix (shared B is not valid when
    // a block spans a group boundary).
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {

        // Cooperative load of the A tile (row = threadIdx.y, col = +k)
        int col_A = k_tile + threadIdx.x;
        s_A[threadIdx.y][threadIdx.x] =
            (col_A < K) ? A[(size_t)global_row * K + col_A] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            const int kk = k_tile + k;
            const float bval =
                (kk < K) ? B_group[(size_t)kk * N + global_col] : 0.0f;
            sum += s_A[threadIdx.y][k] * bval;
        }

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
    grouped_gemm_naive_kernel<<<numBlocks, threadsPerBlock>>>(A, B, group_offsets, C, G, M_total, K, N);
}
