#include <cuda_runtime.h>

#define TILE_SIZE 32

// Tiled shared-memory matrix multiplication: C = A * B
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE + 1]; // +1 padding prevents shared memory bank conflicts

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P) {
    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    // Base case: P == 1 requires only a direct copy
    if (P == 1) {
        cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice);
        return;
    }

    // Allocate temporary device buffers for exponentiation by squaring
    float *d_base, *d_temp;
    cudaMalloc(&d_base, bytes);
    cudaMalloc(&d_temp, bytes);

    cudaMemcpy(d_base, input, bytes, cudaMemcpyDeviceToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    bool has_result = false;
    int exp = P;

    while (exp > 0) {
        if (exp & 1) {
            if (!has_result) {
                cudaMemcpy(output, d_base, bytes, cudaMemcpyDeviceToDevice);
                has_result = true;
            } else {
                matmul_tiled<<<grid, block>>>(output, d_base, d_temp, N);
                cudaMemcpy(output, d_temp, bytes, cudaMemcpyDeviceToDevice);
            }
        }
        exp >>= 1;
        if (exp > 0) {
            matmul_tiled<<<grid, block>>>(d_base, d_base, d_temp, N);
            cudaMemcpy(d_base, d_temp, bytes, cudaMemcpyDeviceToDevice);
        }
    }

    cudaFree(d_base);
    cudaFree(d_temp);
}
