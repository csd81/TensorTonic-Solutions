#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Phase 1: Vectorized grid-stride loop using float4
    // Each thread processes 4 elements (16 bytes) per instruction
    int vec_limit = N / 4;
    const float4* in4 = reinterpret_cast<const float4*>(input);
    float4* out4 = reinterpret_cast<float4*>(output);

    for (int i = idx; i < vec_limit; i += stride) {
        float4 v = in4[i];
        float4 r;
        // The hardware intrinsic tanhf is much faster than explicit expf() math
        r.x = tanhf(v.x);
        r.y = tanhf(v.y);
        r.z = tanhf(v.z);
        r.w = tanhf(v.w);
        out4[i] = r;
    }

    // Phase 2: Scalar tail cleanup
    // Handles the remaining 1 to 3 elements if N is not a multiple of 4
    int tail_start = vec_limit * 4;
    int tail_count = N % 4;
    
    // We only need the first few globally indexed threads to finish the tail
    if (idx < tail_count) {
        output[tail_start + idx] = tanhf(input[tail_start + idx]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    
    // We compute the number of thread blocks based on vectorized elements
    int vec_elements = N / 4 + (N % 4 != 0);
    int blocks = (vec_elements + threads - 1) / threads;
    
    // Cap blocks to fully utilize streaming multiprocessors (e.g., ~512 blocks) 
    // and rely on the grid-stride loop for larger N, keeping occupancy high 
    // without overflowing launch queues.
    if (blocks > 512) {
        blocks = 512;
    }

    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}