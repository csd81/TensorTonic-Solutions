#include <cuda_runtime.h>

__global__ void __launch_bounds__(256, 4) swish_kernel(const float* __restrict__ input, 
                                                      float* __restrict__ output, 
                                                      int N) {
    // Process 8 floats (32 bytes = 2x float4) per iteration for optimal MLP on T4
    const int total_vec8 = N / 8;
    const float4* in_v4 = reinterpret_cast<const float4*>(input);
    float4* out_v4 = reinterpret_cast<float4*>(output);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_vec8; i += stride) {
        // Dual 128-bit loads upfront (LDG.E.128)
        float4 v0 = in_v4[i * 2 + 0];
        float4 v1 = in_v4[i * 2 + 1];

        // SFU pipeline batch: calculate all exponentials together
        float e0 = __expf(-v0.x);
        float e1 = __expf(-v0.y);
        float e2 = __expf(-v0.z);
        float e3 = __expf(-v0.w);

        float e4 = __expf(-v1.x);
        float e5 = __expf(-v1.y);
        float e6 = __expf(-v1.z);
        float e7 = __expf(-v1.w);

        // ALU pipeline: fast reciprocal intrinsic (__frcp_rn) + multiply
        float4 r0, r1;
        r0.x = v0.x * __frcp_rn(1.0f + e0);
        r0.y = v0.y * __frcp_rn(1.0f + e1);
        r0.z = v0.z * __frcp_rn(1.0f + e2);
        r0.w = v0.w * __frcp_rn(1.0f + e3);

        r1.x = v1.x * __frcp_rn(1.0f + e4);
        r1.y = v1.y * __frcp_rn(1.0f + e5);
        r1.z = v1.z * __frcp_rn(1.0f + e6);
        r1.w = v1.w * __frcp_rn(1.0f + e7);

        // Dual 128-bit stores (STG.E.128)
        out_v4[i * 2 + 0] = r0;
        out_v4[i * 2 + 1] = r1;
    }

    // Scalar cleanup loop for remaining unaligned elements
    int tail_start = total_vec8 * 8;
    for (int i = tail_start + idx; i < N; i += stride) {
        float x = input[i];
        output[i] = x * __frcp_rn(1.0f + __expf(-x));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    constexpr int threads = 256;
    
    // T4 has 40 SMs, max 1024 resident threads per SM -> 4 blocks of 256 threads per SM
    // 40 SMs * 4 blocks = 160 blocks to completely populate all SMs in a single wave
    constexpr int max_blocks = 160;
    int needed_blocks = (N + (threads * 8) - 1) / (threads * 8);
    int blocks = (needed_blocks < max_blocks && needed_blocks > 0) ? needed_blocks : max_blocks;

    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}