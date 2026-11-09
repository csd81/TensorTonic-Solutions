#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>

// CUDA kernel to perform 2D max pooling
__global__ void max_pool2d_kernel(const float* __restrict__ input, float* __restrict__ output,
                                   int N, int C, int H, int W,
                                   int H_out, int W_out,
                                   int kernel_size, int stride, int padding) {
    // Each thread processes one element in the output tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;
    
    if (idx >= total_elements) return;

    // Decompose the linear thread index into 4D tensor coordinates
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int n     = idx / (W_out * H_out * C);

    // Calculate the starting position of the pooling window in the input feature map
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float max_val = -FLT_MAX;

    // Linear offset to the beginning of the current channel (n, c)
    int channel_offset = ((n * C) + c) * H * W;

    // Iterate over the local pooling window
    for (int kh = 0; kh < kernel_size; ++kh) {
        int cur_h = h_start + kh;
        if (cur_h >= 0 && cur_h < H) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int cur_w = w_start + kw;
                if (cur_w >= 0 && cur_w < W) {
                    int input_idx = channel_offset + cur_h * W + cur_w;
                    float val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
    }

    // Write the computed max value back to the output tensor
    output[idx] = max_val;
}

extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    // Calculate output dimensions
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    int total_elements = N * C * H_out * W_out;
    if (total_elements <= 0) return;

    // Configure the execution configuration
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch the CUDA kernel
    max_pool2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding
    );

    // Synchronize to ensure completion before returning to host
    cudaDeviceSynchronize();
}
