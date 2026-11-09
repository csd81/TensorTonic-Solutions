#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// CUDA Kernel for W4A16 Matrix Multiplication
__global__ void w4a16_gemm_kernel(
    const __half* __restrict__ x, 
    const uint8_t* __restrict__ w_q, 
    const __half* __restrict__ scales, 
    __half* __restrict__ y, 
    int M, int N, int K, int group_size) 
{
    // Global row (M dimension) and column (N dimension) indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    
    // Each byte in w_q contains 2 elements along the K dimension.
    // Total number of bytes per row of weights is K / 2.
    int num_bytes_k = K / 2;

    // Loop over the K dimension in blocks of 2 elements (1 byte from w_q)
    for (int byte_k = 0; byte_k < num_bytes_k; ++byte_k) {
        // Map byte index back to actual K indices
        int k0 = byte_k * 2;
        int k1 = k0 + 1;

        // Fetch packed INT4 byte for weight matrix row `col` and column `byte_k`
        uint8_t packed_val = w_q[col * num_bytes_k + byte_k];

        // Unpack high nibble (w[n, 2i]) and low nibble (w[n, 2i+1])
        int8_t w0_int = (int8_t)((packed_val >> 4) & 0x0F) - 8;
        int8_t w1_int = (int8_t)(packed_val & 0x0F) - 8;

        // Retrieve group scales matching the current K elements
        __half scale0 = scales[col * (K / group_size) + (k0 / group_size)];
        __half scale1 = scales[col * (K / group_size) + (k1 / group_size)];

        // Dequantize weights to float
        float w0 = (float)w0_int * __half2float(scale0);
        float w1 = (float)w1_int * __half2float(scale1);

        // Fetch activations corresponding to row `row` and accumulate the dot product
        float x0 = __half2float(x[row * K + k0]);
        float x1 = __half2float(x[row * K + k1]);

        acc += x0 * w0 + x1 * w1;
    }

    // Write out final computation back to global memory in float16
    y[row * N + col] = __float2half(acc);
}

extern "C" void solve(
    const __half* x, const uint8_t* w_q, const __half* scales, __half* y, 
    int M, int N, int K, int group_size) 
{
    // Define 2D block dimensions optimizing layout configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    w4a16_gemm_kernel<<<numBlocks, threadsPerBlock>>>(x, w_q, scales, y, M, N, K, group_size);
}
