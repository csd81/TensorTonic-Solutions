#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    // Calculate the output dimensions
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    
    // Calculate the row and column of the output element this thread is responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure the thread maps to a valid output element
    if (row < outH && col < outW) {
        float sum = 0.0f;
        
        // Compute the cross-correlation dot product for the current spatial position
        for (int i = 0; i < kH; ++i) {
            int in_row = row + i;
            for (int j = 0; j < kW; ++j) {
                int in_col = col + j;
                
                // Read from row-major flat arrays
                float in_val = input[in_row * W + in_col];
                float k_val  = kernel[i * kW + j];
                
                sum += in_val * k_val;
            }
        }
        
        // Write out to the calculated global output index
        output[row * outW + col] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    
    conv2d_kernel<<<blocks, threads>>>(input, kernel, output, H, W, kH, kW);
    cudaDeviceSynchronize();
}