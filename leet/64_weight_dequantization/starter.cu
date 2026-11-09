#include <cuda_runtime.h>

// CUDA kernel to perform matrix dequantization
__global__ void dequantize_kernel(const float* __restrict__ X, 
                                 const float* __restrict__ S, 
                                 float* __restrict__ Y, 
                                 int M, int N, int TILE_SIZE,
                                 int S_cols) {
    // Compute the global row and column indices for the current thread
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // Check boundary conditions to avoid out-of-bounds memory access
    if (i < M && j < N) {
        // Calculate the corresponding tile row and column indices
        int row = i / TILE_SIZE;
        int col = j / TILE_SIZE;

        // Compute the flat index for the 2D matrices
        int x_idx = i * N + j;
        int s_idx = row * S_cols + col;

        // Perform the dequantization operation
        Y[x_idx] = X[x_idx] * S[s_idx];
    }
}

extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    // Define block dimensions (2D configuration for optimal thread occupancy)
    dim3 threadsPerBlock(16, 16);

    // Calculate grid dimensions to cover the entire matrix
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Calculate the number of columns in the scale matrix S
    int S_cols = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Launch the CUDA kernel
    dequantize_kernel<<<numBlocks, threadsPerBlock>>>(X, S, Y, M, N, TILE_SIZE, S_cols);

    // Ensure the kernel completes execution before returning
    cudaDeviceSynchronize();
}
