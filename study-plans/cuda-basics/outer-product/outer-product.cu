#include <cuda_runtime.h>

__global__ void outer_product_kernel(const float* __restrict__ a, 
                                     const float* __restrict__ b, 
                                     float* __restrict__ C, 
                                     int M, int N) {
    __shared__ float s_a[16];
    __shared__ float s_b[16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j = blockIdx.x * 16 + tx;
    int i = blockIdx.y * 16 + ty;

    // Load slices cooperatively (only 1 thread per row/col loads)
    if (tx == 0) {
        s_a[ty] = (i < M) ? a[i] : 0.0f;
    }
    if (ty == 0) {
        s_b[tx] = (j < N) ? b[j] : 0.0f;
    }

    __syncthreads();

    if (i < M && j < N) {
        // s_a is broadcast across all 16 threads in the row (0 conflicts)
        // s_b is accessed sequentially across banks (0 conflicts)
        C[i * N + j] = s_a[ty] * s_b[tx];
    }
}

extern "C" void solve(const float* a, const float* b, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    outer_product_kernel<<<blocks, threads>>>(a, b, C, M, N);
    cudaDeviceSynchronize();
}