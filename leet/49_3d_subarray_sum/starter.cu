#include <cuda_runtime.h>

__global__ void sum3d_kernel(const int* __restrict__ input, int* __restrict__ output,
                             int M, int K,
                             int S_DEP, int E_DEP,
                             int S_ROW, int E_ROW,
                             int S_COL, int E_COL) {
    int dep_len = E_DEP - S_DEP + 1;
    int row_len = E_ROW - S_ROW + 1;
    int col_len = E_COL - S_COL + 1;

    long long total_elements = static_cast<long long>(dep_len) * row_len * col_len;
    long long plane_size = static_cast<long long>(row_len) * col_len;
    long long stride_dep = static_cast<long long>(M) * K;

    int local_sum = 0;

    // Grid-stride loop over the flattened 3D bounding box
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        
        int d = idx / plane_size;
        int rem = idx % plane_size;
        int r = rem / col_len;
        int c = rem % col_len;

        long long global_idx = (S_DEP + d) * stride_dep + (S_ROW + r) * K + (S_COL + c);
        local_sum += input[global_idx];
    }

    // 1. Warp-level reduction using shuffle instructions
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // 2. Store each warp's sum into shared memory
    __shared__ int warp_sums[32]; // Max 1024 threads per block -> max 32 warps
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // 3. First warp reduces the partial sums from shared memory
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? warp_sums[lane] : 0;

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }

        // 4. Thread 0 atomically accumulates the block result into global output
        if (lane == 0) {
            atomicAdd(output, local_sum);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP,
                      int S_ROW, int E_ROW, int S_COL, int E_COL) {
    // Initialize output to 0 before accumulating with atomicAdd
    cudaMemset(output, 0, sizeof(int));

    long long total_elements = static_cast<long long>(E_DEP - S_DEP + 1) *
                               (E_ROW - S_ROW + 1) *
                               (E_COL - S_COL + 1);

    if (total_elements <= 0) return;

    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;
    // Cap grid size to maximize occupancy without excessive atomic contention
    if (numBlocks > 1024) {
        numBlocks = 1024;
    }

    sum3d_kernel<<<numBlocks, blockSize>>>(input, output, M, K, S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL);
}
