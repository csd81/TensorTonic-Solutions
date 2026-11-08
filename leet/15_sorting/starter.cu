#include <cuda_runtime.h>
#include <float.h>

// Kernel to copy input data and pad remaining elements with +INFINITY
__global__ void pad_and_copy(const float* __restrict__ src, float* __restrict__ dst, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        dst[idx] = (idx < N) ? src[idx] : INFINITY;
    }
}

// Kernel for a single compare-and-swap step in Bitonic Sort
__global__ void bitonic_sort_step(float* __restrict__ data, int j, int k, int M) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    unsigned int ixj = i ^ j;

    // Only the thread with the lower index performs the comparison and swap
    if (ixj > i) {
        float val_i = data[i];
        float val_ixj = data[ixj];

        // Determine sorting direction for the current bitonic sequence
        bool ascending = ((i & k) == 0);

        if ((ascending && val_i > val_ixj) || (!ascending && val_i < val_ixj)) {
            data[i] = val_ixj;
            data[ixj] = val_i;
        }
    }
}

// data is device pointer
extern "C" void solve(float* data, int N) {
    if (N <= 1) return;

    // Find the smallest power of 2 >= N
    int M = 1;
    while (M < N) {
        M <<= 1;
    }

    int threads = 256;
    int blocks = (M + threads - 1) / threads;

    // If N is already a power of 2, we can sort directly in-place without extra memory
    if (M == N) {
        for (int k = 2; k <= M; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                bitonic_sort_step<<<blocks, threads>>>(data, j, k, M);
            }
        }
        cudaDeviceSynchronize();
        return;
    }

    // Allocate padded device buffer
    float* d_padded = nullptr;
    cudaMalloc(&d_padded, M * sizeof(float));

    // Copy original data and pad with INFINITY
    pad_and_copy<<<blocks, threads>>>(data, d_padded, N, M);

    // Execute Bitonic Sort
    for (int k = 2; k <= M; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<blocks, threads>>>(d_padded, j, k, M);
        }
    }

    // Copy the first N sorted elements back to the input array
    cudaMemcpy(data, d_padded, N * sizeof(float), cudaMemcpyDeviceToDevice);

    // Clean up temporary memory
    cudaFree(d_padded);
    cudaDeviceSynchronize();
}
