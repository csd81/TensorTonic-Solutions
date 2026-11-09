#include <cuda_runtime.h>

// CUDA Kernel to calculate the 5-point Jacobi stencil iteration
__global__ void jacobi_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    // Compute the global row and column indices for the current thread
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to prevent out-of-bounds memory access
    if (r >= rows || c >= cols) {
        return;
    }

    // Boundary cells (first/last row and column) are copied unchanged
    if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1) {
        output[r * cols + c] = input[r * cols + c];
    } else {
        // Interior cells are computed as the average of their four cardinal neighbors
        float top    = input[(r - 1) * cols + c];
        float bottom = input[(r + 1) * cols + c];
        float left   = input[r * cols + (c - 1)];
        float right  = input[r * cols + (c + 1)];
        
        output[r * cols + c] = 0.25f * (top + bottom + left + right);
    }
}

// Host-side wrapper function
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    // 16x16 or 32x32 block sizes are optimal for hiding memory latency and maximizing warp efficiency
    dim3 threadsPerBlock(32, 32);
    
    // Calculate grid dimensions to fully cover the input matrix
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel on the default stream
    jacobi_kernel<<<numBlocks, threadsPerBlock>>>(input, output, rows, cols);
    
    // Ensure host waits for kernel execution to complete safely
    cudaDeviceSynchronize();
}
