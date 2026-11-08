#include <cuda_runtime.h>

__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void sum2DSubarrayKernel(const int* __restrict__ input, int* __restrict__ output,
                                   int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int subRows = E_ROW - S_ROW + 1;
    int subCols = E_COL - S_COL + 1;
    long long totalElements = static_cast<long long>(subRows) * subCols;

    int localSum = 0;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

    // Grid-stride loop over the target submatrix elements
    for (long long i = idx; i < totalElements; i += stride) {
        int r = S_ROW + static_cast<int>(i / subCols);
        int c = S_COL + static_cast<int>(i % subCols);
        localSum += input[static_cast<long long>(r) * M + c];
    }

    // 1. Reduce within each warp
    localSum = warpReduceSum(localSum);

    // 2. Store warp sums into shared memory
    __shared__ int shared[32]; // Max 1024 threads per block -> max 32 warps
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    if (lane == 0) {
        shared[wid] = localSum;
    }
    __syncthreads();

    // 3. First warp reduces the partial warp sums
    localSum = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) {
        localSum = warpReduceSum(localSum);
        // 4. Add block sum to global output
        if (lane == 0 && localSum != 0) {
            atomicAdd(output, localSum);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
    // Initialize output on the device to 0
    cudaMemset(output, 0, sizeof(int));

    long long totalElements = static_cast<long long>(E_ROW - S_ROW + 1) * (E_COL - S_COL + 1);
    if (totalElements <= 0) return;

    int threadsPerBlock = 256;
    // Cap blocks to avoid launch overhead on small subarrays while saturating SMs on large ones
    int maxBlocks = 1024; 
    int blocksPerGrid = static_cast<int>((totalElements + threadsPerBlock - 1) / threadsPerBlock);
    if (blocksPerGrid > maxBlocks) {
        blocksPerGrid = maxBlocks;
    }

    sum2DSubarrayKernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, M, S_ROW, E_ROW, S_COL, E_COL
    );
}
