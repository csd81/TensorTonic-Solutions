#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Optimized block dimension (32x32 = 1024 threads per block)
#define BLOCK_SIZE 32

__global__ void floyd_warshall_kernel(float* output, int N, int k) {
    // Calculate global row and column indices for this thread
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check for the matrix dimensions
    if (i < N && j < N) {
        // Load the current known shortest distance between i and j
        float current_dist = output[i * N + j];
        
        // Load the distance from i to intermediate k, and k to j
        float ik = output[i * N + k];
        float kj = output[k * N + j];

        // Only update if a valid path exists through k to avoid adding infinity
        if (ik != INFINITY && kj != INFINITY) {
            float new_dist = ik + kj;
            if (new_dist < current_dist) {
                output[i * N + j] = new_dist;
            }
        }
    }
}

extern "C" void solve(const float* dist, float* output, int N) {
    // Copy the initial state from the distance matrix to the output matrix on the device
    cudaMemcpy(output, dist, sizeof(float) * N * N, cudaMemcpyDeviceToDevice);

    // Configure the 2D grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Sequentially iterate through each intermediate vertex k
    for (int k = 0; k < N; ++k) {
        floyd_warshall_kernel<<<gridSize, blockSize>>>(output, N, k);
        
        // Synchronize threads across iterations to guarantee that all paths 
        // using k-1 are fully updated before evaluating paths using k
        cudaDeviceSynchronize();
    }
}
