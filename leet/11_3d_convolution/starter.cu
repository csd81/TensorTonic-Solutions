#include <cuda_runtime.h>

// Maximum kernel size is 5x5x5 = 125 floats. 
// Constant memory provides fast, cached broadcast reads across warps.
__constant__ float d_kernel[125];

__global__ void conv3d_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              int input_depth, int input_rows, int input_cols,
                              int kernel_depth, int kernel_rows, int kernel_cols,
                              int output_depth, int output_rows, int output_cols) {
    // Map threads to output coordinates (k = col, j = row, i = depth)
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < output_depth && j < output_rows && k < output_cols) {
        float sum = 0.0f;
        
        int input_plane_size = input_rows * input_cols;
        int kernel_plane_size = kernel_rows * kernel_cols;

        // Perform the 3D cross-correlation / convolution sum
        for (int d = 0; d < kernel_depth; ++d) {
            int in_d_offset = (i + d) * input_plane_size;
            int k_d_offset = d * kernel_plane_size;
            
            for (int r = 0; r < kernel_rows; ++r) {
                int in_r_offset = in_d_offset + (j + r) * input_cols;
                int k_r_offset = k_d_offset + r * kernel_cols;
                
                for (int c = 0; c < kernel_cols; ++c) {
                    float in_val = input[in_r_offset + (k + c)];
                    float k_val = d_kernel[k_r_offset + c];
                    sum += in_val * k_val;
                }
            }
        }

        // Write result to row-major, then depth 1D output array
        int out_idx = i * (output_rows * output_cols) + j * output_cols + k;
        output[out_idx] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {
    // Compute output dimensions for "valid" convolution
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    // Copy kernel weights from device global memory to constant memory
    size_t kernel_bytes = kernel_depth * kernel_rows * kernel_cols * sizeof(float);
    cudaMemcpyToSymbol(d_kernel, kernel, kernel_bytes, 0, cudaMemcpyDeviceToDevice);

    // Configure 3D thread block and grid dimensions
    // 16x16x2 = 512 threads per block, aligning well with 128x128 benchmark dimensions
    dim3 threadsPerBlock(16, 16, 2);
    dim3 numBlocks((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (output_depth + threadsPerBlock.z - 1) / threadsPerBlock.z);

    conv3d_kernel<<<numBlocks, threadsPerBlock>>>(
        input, output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        output_depth, output_rows, output_cols
    );
}
