#include <cuda_runtime.h>

__global__ void hadamard_kernel(const float* __restrict__ A, 
                                const float* __restrict__ B, 
                                float* __restrict__ C, 
                                int M, int N) {
    // Thread index mapping (column-walking to ensure coalesced access)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        // Use size_t for the 1D index to prevent 32-bit integer overflow 
        // on large matrices and yield cleaner 64-bit addressing assembly.
        size_t idx = (size_t)i * N + j;
        
        // __ldg() explicitly routes loads through the read-only texture cache.
        // Combined with __restrict__, this maximizes memory throughput on T4.
        C[idx] = __ldg(&A[idx]) * __ldg(&B[idx]);
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    hadamard_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}