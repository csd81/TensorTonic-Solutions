#include <cuda_runtime.h>

// Naive baseline for challenge 133, kept as the starter's shape:
//  * one thread per output element for the base product, K-loop reading W as
//    a column-strided row (0.116 TFLOP/s at the benchmark shape);
//  * the starter's LoRA path, which recomputes x_t . A_a (R*K multiply-adds)
//    for *every output element*, O(T*N*R*K) = 2.7e11 operations.
// Total at T=1024, K=4096, N=4096, R=16, L=8: ~3.74 s.

// Standard base matrix multiplication kernel: y = x * W
__global__ void base_gemm_naive_kernel(const float* __restrict__ x, const float* __restrict__ W, float* __restrict__ y, int T, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < T && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += x[row * K + k] * W[k * N + col];
        }
        y[row * N + col] = sum;
    }
}

// LoRA adjustment kernel: computes and adds scale * (x_t * A_a) * B_a to y
__global__ void lora_adapter_naive_kernel(const float* __restrict__ x, 
                                    const float* __restrict__ A, 
                                    const float* __restrict__ B,
                                    const int* __restrict__ adapter_ids, 
                                    float* __restrict__ y, 
                                    int T, int K, int N, int R, float scale) {
    int t = blockIdx.y; // Each block along y handles one token
    int n = blockIdx.x * blockDim.x + threadIdx.x; // Column mapping for N

    if (t >= T || n >= N) return;

    int a = adapter_ids[t];
    if (a < 0) return; // Skip if token uses no adapter (-1)

    // Pointers to the adapter slices for the active token
    const float* A_a = A + (a * K * R);
    const float* B_a = B + (a * R * N);
    const float* x_t = x + (t * K);

    // Compute intermediate vector: temp[r] = x_t * A_a (Dimension: 1 x R)
    // Since R <= 64, we can comfortably cache this inside a small dynamic buffer in registers/shared memory per thread or cooperatively.
    // Given the block-level structure, let's accumulate it directly inside the thread sequentially for maximum clarity and robustness.
    float lora_sum = 0.0f;

    for (int r = 0; r < R; ++r) {
        // Compute x_t * A_a[:, r]
        float xr = 0.0f;
        for (int k = 0; k < K; ++k) {
            xr += x_t[k] * A_a[k * R + r];
        }
        // Multiply by B_a[r, n]
        lora_sum += xr * B_a[r * N + n];
    }

    // Accumulate the final scaled low-rank update into y
    y[t * N + n] += scale * lora_sum;
}

extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      const int* adapter_ids, float* y, int T, int K, int N, int R, int L,
                      float scale) {
    // 1. Compute Base Layer: y = X * W
    dim3 block_base(16, 16);
    dim3 grid_base((N + block_base.x - 1) / block_base.x, (T + block_base.y - 1) / block_base.y);
    base_gemm_naive_kernel<<<grid_base, block_base>>>(x, W, y, T, K, N);

    // 2. Compute and accumulate the LoRA updates
    dim3 block_lora(256);
    dim3 grid_lora((N + block_lora.x - 1) / block_lora.x, T);
    lora_adapter_naive_kernel<<<grid_lora, block_lora>>>(x, A, B, adapter_ids, y, T, K, N, R, scale);
}
