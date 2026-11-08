#include <cuda_runtime.h>

#define BLOCK_SIZE 512

// Computes an inclusive prefix sum within a single warp (32 threads)
__device__ __forceinline__ float warp_inclusive_scan(float val) {
    unsigned int mask = 0xffffffff;
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(mask, val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val += n;
        }
    }
    return val;
}

// Step 1: Block-level inclusive scan + record total block sums
__global__ void block_scan_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  float* __restrict__ block_sums,
                                  int N) {
    __shared__ float warp_sums[BLOCK_SIZE / 32];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    // Load input (pad out-of-bounds threads with 0.0f)
    float val = (gid < N) ? input[gid] : 0.0f;

    // 1. Inclusive scan within each warp
    val = warp_inclusive_scan(val);

    // 2. Store the total sum of each warp into shared memory
    if (lane == 31) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // 3. First warp scans the warp sums (supports up to 32 warps = 1024 threads)
    if (warp_id == 0) {
        float w_val = (lane < (BLOCK_SIZE / 32)) ? warp_sums[lane] : 0.0f;
        w_val = warp_inclusive_scan(w_val);
        warp_sums[lane] = w_val;
    }
    __syncthreads();

    // 4. Add the scanned sum of preceding warps to each thread's value
    if (warp_id > 0) {
        val += warp_sums[warp_id - 1];
    }

    // Write inclusive prefix sum to global output
    if (gid < N) {
        output[gid] = val;
    }

    // Last thread in the block writes the total block sum for hierarchical propagation
    if (tid == BLOCK_SIZE - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = val;
    }
}

// Step 3: Add scanned block offsets back to each element in the block
__global__ void add_block_offsets_kernel(float* __restrict__ output,
                                         const float* __restrict__ scanned_block_sums,
                                         int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && gid < N) {
        output[gid] += scanned_block_sums[blockIdx.x - 1];
    }
}

// Recursive host helper to handle arbitrary array sizes
void recursive_scan(const float* d_in, float* d_out, int N) {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (num_blocks == 1) {
        block_scan_kernel<<<1, BLOCK_SIZE>>>(d_in, d_out, nullptr, N);
        return;
    }

    float* d_block_sums = nullptr;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));

    // 1. Scan each block locally and save block totals
    block_scan_kernel<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, d_block_sums, N);

    // 2. Recursively scan the array of block totals in-place
    recursive_scan(d_block_sums, d_block_sums, num_blocks);

    // 3. Propagate scanned block totals to subsequent blocks
    add_block_offsets_kernel<<<num_blocks, BLOCK_SIZE>>>(d_out, d_block_sums, N);

    cudaFree(d_block_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;
    recursive_scan(input, output, N);
}
