#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* __restrict__ input, int* __restrict__ histogram, int N, int num_bins) {
    // Dynamically allocated shared memory for block-local histogram
    extern __shared__ int s_hist[];

    // 1. Initialize shared memory bins to 0
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // 2. Grid-stride loop to process input elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Vectorized loads (int4) for aligned memory access
    int n_vec = N / 4;
    const int4* input_vec = reinterpret_cast<const int4*>(input);

    for (int i = idx; i < n_vec; i += stride) {
        int4 val = input_vec[i];
        atomicAdd(&s_hist[val.x], 1);
        atomicAdd(&s_hist[val.y], 1);
        atomicAdd(&s_hist[val.z], 1);
        atomicAdd(&s_hist[val.w], 1);
    }

    // Handle remaining tail elements
    for (int i = n_vec * 4 + idx; i < N; i += stride) {
        atomicAdd(&s_hist[input[i]], 1);
    }
    __syncthreads();

    // 3. Accumulate block-local counts into global memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        int count = s_hist[i];
        if (count > 0) {
            atomicAdd(&histogram[i], count);
        }
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    // Ensure output histogram is zero-initialized
    cudaMemset(histogram, 0, num_bins * sizeof(int));

    int threads_per_block = 256;
    
    // Query SM count to launch an optimal grid size for grid-stride loops
    int device_id = 0;
    int num_sms = 0;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);

    // 32 blocks per SM provides good latency hiding while limiting global atomic overhead
    int blocks_per_grid = num_sms * 32;
    int max_needed_blocks = (N + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > max_needed_blocks) {
        blocks_per_grid = max_needed_blocks;
    }

    size_t shared_mem_size = num_bins * sizeof(int);

    histogram_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        input, histogram, N, num_bins
    );
}
