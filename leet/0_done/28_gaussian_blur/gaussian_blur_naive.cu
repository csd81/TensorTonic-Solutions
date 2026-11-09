#include <cuda_runtime.h>

// Naive baseline for challenge 28, kept as the starter's shape: a fixed
// 36x36 shared tile (sized for the maximum radius 10 whatever the kernel
// is), 16x16 threads, one output per thread, and one scalar shared load per
// multiply-add with runtime kernel-size loops that cannot unroll.  Measures
// 212 us at 512x512 with a 7x7 kernel (the copy floor is ~32 us).

#define BLOCK_SIZE 16
#define MAX_RADIUS 10  // Max kernel size is 21 -> radius = 21 / 2 = 10
#define TILE_DIM (BLOCK_SIZE + 2 * MAX_RADIUS)

__global__ void gaussian_blur_naive_kernel(const float* __restrict__ input,
                                     const float* __restrict__ kernel,
                                     float* __restrict__ output,
                                     int input_rows, int input_cols,
                                     int kernel_rows, int kernel_cols) {
    // Shared memory tile including halo for zero-padding/neighbor access
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int pad_h = kernel_rows / 2;
    int pad_w = kernel_cols / 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Top-left corner of the tile in global input coordinates (including halo)
    int tile_start_row = blockIdx.y * BLOCK_SIZE - pad_h;
    int tile_start_col = blockIdx.x * BLOCK_SIZE - pad_w;

    int tile_rows = BLOCK_SIZE + 2 * pad_h;
    int tile_cols = BLOCK_SIZE + 2 * pad_w;

    // Cooperatively load the input tile + halo into shared memory
    for (int i = ty; i < tile_rows; i += BLOCK_SIZE) {
        for (int j = tx; j < tile_cols; j += BLOCK_SIZE) {
            int global_r = tile_start_row + i;
            int global_c = tile_start_col + j;

            // Apply zero-padding for out-of-bounds indices
            if (global_r >= 0 && global_r < input_rows &&
                global_c >= 0 && global_c < input_cols) {
                tile[i][j] = input[global_r * input_cols + global_c];
            } else {
                tile[i][j] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Global output coordinates for this thread
    int out_r = blockIdx.y * BLOCK_SIZE + ty;
    int out_c = blockIdx.x * BLOCK_SIZE + tx;

    if (out_r < input_rows && out_c < input_cols) {
        float sum = 0.0f;
        for (int kr = 0; kr < kernel_rows; ++kr) {
            for (int kc = 0; kc < kernel_cols; ++kc) {
                sum += tile[ty + kr][tx + kc] * kernel[kr * kernel_cols + kc];
            }
        }
        output[out_r * input_cols + out_c] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((input_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (input_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gaussian_blur_naive_kernel<<<gridDim, blockDim>>>(
        input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
