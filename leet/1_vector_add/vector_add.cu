#include <cuda_runtime.h>
__global__ void vector_add(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
