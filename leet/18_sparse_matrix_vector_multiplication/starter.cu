#include <cuda_runtime.h>

__global__ void spmv_dense_warp_kernel(const float* __restrict__ A, 
                                     const float* __restrict__ x, 
                                     float* __restrict__ y, 
                                     int M, int N) {
    // Each warp (32 threads) processes one row of matrix A
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= M) return;

    int lane = threadIdx.x % 32;
    const float* row_A = A + (size_t)row * N;

    float sum = 0.0f;

    // 1. Vectorized float4 loads for coalesced 128-bit memory transactions
    // Check 16-byte alignment of the row pointer before casting to float4
    int j = lane * 4;
    if (((uintptr_t)row_A & 15) == 0) {
        int N_vec = N & ~3; // Largest multiple of 4 <= N
        for (; j < N_vec; j += 128) { // 32 threads * 4 floats = 128 elements per step
            float4 a_val = *reinterpret_cast<const float4*>(&row_A[j]);
            
            // Skip x lookups if the entire float4 block is zero
            if (a_val.x != 0.0f) sum += a_val.x * __ldg(&x[j]);
            if (a_val.y != 0.0f) sum += a_val.y * __ldg(&x[j + 1]);
            if (a_val.z != 0.0f) sum += a_val.z * __ldg(&x[j + 2]);
            if (a_val.w != 0.0f) sum += a_val.w * __ldg(&x[j + 3]);
        }
        // Reset j for remaining tail elements (when N is not divisible by 4)
        j = N_vec + lane;
    } else {
        // Fallback if row pointer is not 16-byte aligned
        j = lane;
    }

    // 2. Scalar loop for remaining tail elements (or unaligned rows)
    for (; j < N; j += 32) {
        float a_val = row_A[j];
        if (a_val != 0.0f) {
            sum += a_val * __ldg(&x[j]);
        }
    }

    // 3. Fast warp-level tree reduction using shuffle instructions
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 writes the final dot product for this row
    if (lane == 0) {
        y[row] = sum;
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    if (M <= 0 || N <= 0) return;

    // 256 threads per block = 8 warps per block (processes 8 rows per block)
    const int threadsPerBlock = 256;
    const int warpsPerBlock = threadsPerBlock / 32;
    const int blocksPerGrid = (M + warpsPerBlock - 1) / warpsPerBlock;

    spmv_dense_warp_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, x, y, M, N);
}
