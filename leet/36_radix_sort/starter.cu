#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_BUCKETS 16

// Step 1: Compute per-block histograms for the current 4-bit digit
__global__ void histogram_kernel(const unsigned int* __restrict__ data,
                                 unsigned int* __restrict__ block_hist,
                                 int N, int shift, int grid_size) {
    __shared__ unsigned int local_hist[RADIX_BUCKETS];

    if (threadIdx.x < RADIX_BUCKETS) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        unsigned int digit = (data[idx] >> shift) & 0xF;
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // Write column-major so same-digit counts across blocks are contiguous in global memory
    if (threadIdx.x < RADIX_BUCKETS) {
        block_hist[threadIdx.x * grid_size + blockIdx.x] = local_hist[threadIdx.x];
    }
}

// Step 2a: Blelloch exclusive scan within each block
__global__ void block_scan_kernel(unsigned int* __restrict__ data,
                                  unsigned int* __restrict__ block_sums,
                                  int total_elements) {
    __shared__ unsigned int temp[BLOCK_SIZE * 2];
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * (BLOCK_SIZE * 2);

    int idx1 = block_offset + tid;
    int idx2 = block_offset + tid + BLOCK_SIZE;

    temp[tid] = (idx1 < total_elements) ? data[idx1] : 0;
    temp[tid + BLOCK_SIZE] = (idx2 < total_elements) ? data[idx2] : 0;

    // Up-sweep (reduce) phase
    int offset = 1;
    for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Save block total and clear last element for exclusive scan
    if (tid == 0) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = temp[BLOCK_SIZE * 2 - 1];
        }
        temp[BLOCK_SIZE * 2 - 1] = 0;
    }

    // Down-sweep phase
    for (int d = 1; d < BLOCK_SIZE * 2; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (idx1 < total_elements) data[idx1] = temp[tid];
    if (idx2 < total_elements) data[idx2] = temp[tid + BLOCK_SIZE];
}

// Step 2b: Add scanned block sums back to each block
__global__ void add_block_sums_kernel(unsigned int* __restrict__ data,
                                      const unsigned int* __restrict__ block_sums,
                                      int total_elements) {
    int block_offset = blockIdx.x * (BLOCK_SIZE * 2);
    unsigned int add_val = block_sums[blockIdx.x];

    int idx1 = block_offset + threadIdx.x;
    int idx2 = block_offset + threadIdx.x + BLOCK_SIZE;

    if (idx1 < total_elements) data[idx1] += add_val;
    if (idx2 < total_elements) data[idx2] += add_val;
}

// Recursive helper for arbitrary-length GPU exclusive prefix sum
void gpu_exclusive_scan(unsigned int* d_data, int num_elements) {
    if (num_elements <= 1) {
        if (num_elements == 1) cudaMemset(d_data, 0, sizeof(unsigned int));
        return;
    }

    int elements_per_block = BLOCK_SIZE * 2;
    int num_blocks = (num_elements + elements_per_block - 1) / elements_per_block;

    if (num_blocks == 1) {
        block_scan_kernel<<<1, BLOCK_SIZE>>>(d_data, nullptr, num_elements);
    } else {
        unsigned int* d_block_sums = nullptr;
        cudaMalloc(&d_block_sums, num_blocks * sizeof(unsigned int));

        block_scan_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_block_sums, num_elements);
        gpu_exclusive_scan(d_block_sums, num_blocks);
        add_block_sums_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_block_sums, num_elements);

        cudaFree(d_block_sums);
    }
}

// Step 3: Stable local ranking via warp-level bitmasking + scatter to global offsets
__global__ void scatter_kernel(const unsigned int* __restrict__ src,
                               unsigned int* __restrict__ dst,
                               const unsigned int* __restrict__ global_offsets,
                               int N, int shift, int grid_size) {
    __shared__ unsigned int warp_counts[8][RADIX_BUCKETS]; // 256 threads = 8 warps
    __shared__ unsigned int block_base_offsets[RADIX_BUCKETS];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    // Load global starting offsets for this block's digits
    if (tid < RADIX_BUCKETS) {
        block_base_offsets[tid] = global_offsets[tid * grid_size + blockIdx.x];
    }

    int idx = blockIdx.x * blockDim.x + tid;
    bool valid = (idx < N);
    unsigned int val = valid ? src[idx] : 0;
    unsigned int digit = valid ? ((val >> shift) & 0xF) : 16;

    // Warp-level match to find each thread's rank within its warp for its digit
    unsigned int active_mask = __activemask();
    unsigned int peer_mask = 0;
    for (int d = 0; d < RADIX_BUCKETS; ++d) {
        unsigned int ballot = __ballot_sync(active_mask, digit == d);
        if (digit == d) {
            peer_mask = ballot;
        }
        if (lane_id == 0) {
            warp_counts[warp_id][d] = __popc(ballot);
        }
    }
    __syncthreads();

    // Compute rank within the block by summing earlier warps' counts for the same digit
    if (valid) {
        unsigned int rank_in_warp = __popc(peer_mask & ((1u << lane_id) - 1));
        unsigned int rank_in_block = rank_in_warp;
        for (int w = 0; w < warp_id; ++w) {
            rank_in_block += warp_counts[w][digit];
        }

        unsigned int dest_idx = block_base_offsets[digit] + rank_in_block;
        dst[dest_idx] = val;
    }
}

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {
    if (N <= 0) return;

    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int hist_size = RADIX_BUCKETS * grid_size;

    unsigned int* d_temp = nullptr;
    unsigned int* d_hist = nullptr;

    cudaMalloc(&d_temp, N * sizeof(unsigned int));
    cudaMalloc(&d_hist, hist_size * sizeof(unsigned int));

    // Copy input into d_temp so we can ping-pong cleanly across 8 passes
    // After 8 (an even number) passes: d_temp -> output -> ... -> output
    cudaMemcpy(d_temp, input, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    unsigned int* src = d_temp;
    unsigned int* dst = output;

    // 8 passes * 4 bits = 32 bits
    for (int pass = 0; pass < 8; ++pass) {
        int shift = pass * RADIX_BITS;

        // 1. Count digit occurrences per block
        histogram_kernel<<<grid_size, BLOCK_SIZE>>>(src, d_hist, N, shift, grid_size);

        // 2. Exclusive prefix sum across all (digit, block) counts
        gpu_exclusive_scan(d_hist, hist_size);

        // 3. Scatter elements stably to their sorted positions for this 4-bit pass
        scatter_kernel<<<grid_size, BLOCK_SIZE>>>(src, dst, d_hist, N, shift, grid_size);

        // Swap ping-pong buffers
        unsigned int* swap_tmp = src;
        src = dst;
        dst = swap_tmp;
    }

    // Because 8 is even, the final sorted result lands directly in `output`
    cudaFree(d_temp);
    cudaFree(d_hist);
}
