#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output, 
                              int D, int H, int W, int kD, int kH, int kW) {
    // Determine output dimensions
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    // Map thread coordinates: x -> W (columns), y -> H (rows), z -> D (depth)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column index (width)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row index (height)
    int d = blockIdx.z * blockDim.z + threadIdx.z; // slice index (depth)

    // Boundary check
    if (d < outD && i < outH && j < outW) {
        float sum = 0.0f;

        // Triple-nested loop over filter dimensions
        for (int a = 0; a < kD; ++a) {
            for (int b = 0; b < kH; ++b) {
                for (int c = 0; c < kW; ++c) {
                    int in_idx = (d + a) * H * W + (i + b) * W + (j + c);
                    int k_idx  = a * kH * kW + b * kW + c;
                    sum += input[in_idx] * kernel[k_idx];
                }
            }
        }

        // Linearized output index in row-major order: (d * outH + i) * outW + j
        int out_idx = d * outH * outW + i * outW + j;
        output[out_idx] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, 
                      int D, int H, int W, int kD, int kH, int kW) {
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    dim3 threads(8, 8, 8);
    dim3 blocks((outW + 7) / 8, (outH + 7) / 8, (outD + 7) / 8);

    conv3d_kernel<<<blocks, threads>>>(input, kernel, output, D, H, W, kD, kH, kW);
    cudaDeviceSynchronize();
}