#include <cuda_runtime.h>

#define TILE 32

// Classic shared-memory transpose:
//   - a 32x32 input tile is staged in shared memory (padded to 32x33),
//   - 32x8 threads load/store four elements each (j = 0, 8, 16, 24),
//   - global reads and writes are coalesced along 32 consecutive threads.
__global__ void matrix_transpose_kernel(const float* __restrict__ input,
                                        float* __restrict__ output, int rows,
                                        int cols) {
    __shared__ float tile[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int in_col = blockIdx.x * TILE + tx;
    if (in_col < cols) {
#pragma unroll
        for (int j = 0; j < TILE; j += blockDim.y) {
            const int in_row = blockIdx.y * TILE + ty + j;
            if (in_row < rows) {
                tile[ty + j][tx] = input[in_row * cols + in_col];
            }
        }
    }

    __syncthreads();

    const int out_col = blockIdx.y * TILE + tx;
    if (out_col < rows) {
#pragma unroll
        for (int j = 0; j < TILE; j += blockDim.y) {
            const int out_row = blockIdx.x * TILE + ty + j;
            if (out_row < cols) {
                output[out_row * rows + out_col] = tile[tx][ty + j];
            }
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(32, 8);
    dim3 blocksPerGrid((cols + TILE - 1) / TILE,
                       (rows + TILE - 1) / TILE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
