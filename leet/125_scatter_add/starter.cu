#include <cuda_runtime.h>

__global__ void scatter_add_kernel(
    const float* __restrict__ src,
    const int* __restrict__ index,
    float* __restrict__ output,
    int N,
    int D
) {
    // Map x-dimension to feature dimension D for coalesced memory access
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && d < D) {
        int dst_row = index[i];
        float val = src[static_cast<size_t>(i) * D + d];
        atomicAdd(&output[static_cast<size_t>(dst_row) * D + d], val);
    }
}

// src, index, output are device pointers
extern "C" void solve(const float* src, const int* index, float* output, int N, int D, int M) {
    // Zero-initialize output because its initial contents are undefined
    size_t output_bytes = static_cast<size_t>(M) * D * sizeof(float);
    cudaMemsetAsync(output, 0, output_bytes);

    if (N == 0 || D == 0) {
        return;
    }

    // Use a 2D block layout: 32 threads along D (1 warp) x 8 rows along N = 256 threads
    dim3 block(32, 8);
    dim3 grid(
        (D + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    scatter_add_kernel<<<grid, block>>>(src, index, output, N, D);
}
