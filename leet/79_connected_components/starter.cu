#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel to initialize labels array
__global__ void init_labels(int* labels, int N) {
    int idx = blockIdx.x * blockDim.blockDim.x + threadIdx.x;
    if (idx < N) {
        labels[idx] = idx;
    }
}

// CUDA kernel to perform conditional hooking (linking components)
__global__ void hook_edges(const int* src, const int* dst, int* labels, int M, bool* d_changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        int u = src[idx];
        int v = dst[idx];

        // Find current roots (representative elements)
        int p_u = labels[u];
        int p_v = labels[v];

        if (p_u != p_v) {
            // We want to hook the larger root to the smaller root 
            // to satisfy the minimum vertex ID label requirement.
            int lower = p_u < p_v ? p_u : p_v;
            int higher = p_u > p_v ? p_u : p_v;

            // AtomicMin ensures parallel safety when multiple edges compete to update the same root
            int old = atomicMin(&labels[higher], lower);
            if (old > lower) {
                *d_changed = true;
            }
        }
    }
}

// CUDA kernel for pointer jumping to flatten the trees and accelerate root finding
__global__ void pointer_jump(int* labels, int N, bool* d_changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int p = labels[idx];
        int gp = labels[p];
        if (p != gp) {
            labels[idx] = gp;
            *d_changed = true;
        }
    }
}

extern "C" void solve(const int* src, const int* dst, int* labels, int N, int M) {
    // Thread block configuration
    const int BLOCK_SIZE = 256;
    int num_blocks_N = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_M = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. Initialize each vertex to point to itself
    init_labels<<<num_blocks_N, BLOCK_SIZE>>>(labels, N);
    cudaDeviceSynchronize();

    // Allocate an iteration flag on the device to monitor convergence
    bool h_changed;
    bool* d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    // 2. Iteratively hook edges and flatten paths until no labels change
    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        // Link components through the edges
        hook_edges<<<num_blocks_M, BLOCK_SIZE>>>(src, dst, labels, M, d_changed);
        
        // Shorten paths (pointer jumping)
        pointer_jump<<<num_blocks_N, BLOCK_SIZE>>>(labels, N, d_changed);

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_changed);

    // 3. Final flattening pass to ensure all nodes point directly to their absolute minimum roots
    do {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        pointer_jump<<<num_blocks_N, BLOCK_SIZE>>>(labels, N, d_changed);

        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (h_changed);

    // Free device tracking flag
    cudaFree(d_changed);
}
