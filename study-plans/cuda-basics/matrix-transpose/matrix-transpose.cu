#include <cuda_runtime.h>

#define TILE 16

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    // Stage a 16x16 tile of A through shared memory, padded by one column so
    // the transposed read hits 32 distinct banks instead of one.
    __shared__ float tile[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Load: consecutive tx -> consecutive columns of A, fully coalesced.
    const int i = blockIdx.y * TILE + ty;   // row in A
    const int j = blockIdx.x * TILE + tx;   // column in A
    tile[ty][tx] = (i < M && j < N) ? A[i * N + j] : 0.0f;

    __syncthreads();

    // Store: swap the indices so consecutive tx -> consecutive rows of B,
    // keeping the write coalesced too. The strided access stays in shared
    // memory, where the +1 pad keeps it bank-conflict free.
    const int bi = blockIdx.y * TILE + tx;  // row in B (= column in A)
    const int bj = blockIdx.x * TILE + ty;  // column in B (= row in A)
    if (bi < M && bj < N) {
        B[bj * M + bi] = tile[tx][ty];
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
