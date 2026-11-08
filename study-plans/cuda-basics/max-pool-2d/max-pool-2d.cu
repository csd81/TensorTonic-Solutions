#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_2d_kernel(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    // Calculate the dimensions of the output tensor
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;

    // Calculate the row and column of the output element this thread is responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform bounds checking to ensure threads outside the output range do no work
    if (row < outH && col < outW) {
        // Initialize the running maximum to -FLT_MAX (the identity for max over floats)
        float max_val = -FLT_MAX;
        
        // Compute the starting indices of the pooling window in the input tensor
        int in_row_start = row * sH;
        int in_col_start = col * sW;

        // Iterate over the kH x kW window
        for (int i = 0; i < kH; ++i) {
            for (int j = 0; j < kW; ++j) {
                // Compute the flat index for the input tensor
                int in_idx = (in_row_start + i) * W + (in_col_start + j);
                
                // Read the element and update the running max using the branchless intrinsic
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
        
        // Write the local maximum out to the global memory output tensor once
        int out_idx = row * outW + col;
        output[out_idx] = max_val;
    }
}

extern "C" void solve(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    max_pool_2d_kernel<<<blocks, threads>>>(input, output, H, W, kH, kW, sH, sW);
    cudaDeviceSynchronize();
}