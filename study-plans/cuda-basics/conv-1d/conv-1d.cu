#include <cuda_runtime.h>

__global__ void conv1d_shared_kernel(const float* input, const float* kernel, float* output, int N, int kN) {
    // Dynamic shared memory allocated at launch (size: blockDim.x + kN - 1)
    extern __shared__ float s_input[];

    int tid = threadIdx.x;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. Cooperative Load: Main Tile
    // Every thread loads one element from global memory to shared memory
    if (out_idx < N) {
        s_input[tid] = input[out_idx];
    }

    // 2. Cooperative Load: The Halo
    // The first (kN - 1) threads load the trailing elements needed by the last threads in the block
    if (tid < kN - 1) {
        int halo_idx = out_idx + blockDim.x;
        if (halo_idx < N) {
            s_input[tid + blockDim.x] = input[halo_idx];
        }
    }

    // Barrier: Ensure the entire tile and halo are loaded before any thread computes
    __syncthreads();

    // 3. Compute using Shared Memory
    int outN = N - kN + 1;
    if (out_idx < outN) {
        float sum = 0.0f;
        for (int a = 0; a < kN; ++a) {
            // Read input from fast shared memory, not global memory
            sum += s_input[tid + a] * kernel[a];
        }
        output[out_idx] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int threads = 256;
    dim3 blocks((outN + threads - 1) / threads);
    
    // Shared memory size = (threads + halo size) * 4 bytes
    size_t shared_mem_bytes = (threads + kN - 1) * sizeof(float);
    
    conv1d_shared_kernel<<<blocks, threads, shared_mem_bytes>>>(input, kernel, output, N, kN);
    cudaDeviceSynchronize();
}