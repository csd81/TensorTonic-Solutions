#include <cuda_runtime.h>
#include <limits.h>

// Device kernel to compute the sum of each window of size window_size
__global__ void compute_window_sums(const int* input, int* window_sums, int num_windows, int window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_windows) {
        int current_sum = 0;
        for (int i = 0; i < window_size; ++i) {
            current_sum += input[idx + i];
        }
        window_sums[idx] = current_sum;
    }
}

// Device kernel to find the maximum value using block-level parallel reduction
__global__ void find_max_reduction(const int* window_sums, int num_windows, int* output) {
    // Shared memory for block reduction
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory, defaulting to INT_MIN if out of bounds
    int local_max = (gid < num_windows) ? window_sums[gid] : INT_MIN;
    shared_data[tid] = local_max;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_data[tid + s] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + s];
            }
        }
        __syncthreads();
    }

    // Write the block's maximum safely to the global output
    if (tid == 0) {
        atomicMax(output, shared_data[0]);
    }
}

// Host entry point function
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    // There are exactly (N - window_size + 1) valid window starting positions
    int num_windows = N - window_size + 1;
    if (num_windows <= 0) {
        return;
    }

    // Allocate an intermediate buffer on the GPU device to hold the sum of each window
    int* d_window_sums = nullptr;
    cudaMalloc(&d_window_sums, num_windows * sizeof(int));

    // Initialize output value on the device to INT_MIN to handle negative sums correctly
    int initial_max = INT_MIN;
    cudaMemcpy(output, &initial_max, sizeof(int), cudaMemcpyHostToDevice);

    // Define execution configuration
    int threads_per_block = 256;
    int blocks_per_grid = (num_windows + threads_per_block - 1) / threads_per_block;

    // Step 1: Compute the window sums in parallel
    compute_window_sums<<<blocks_per_grid, threads_per_block>>>(input, d_window_sums, num_windows, window_size);

    // Step 2: Find the maximum value from the window sums using atomicMax reduction
    size_t shared_mem_size = threads_per_block * sizeof(int);
    find_max_reduction<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_window_sums, num_windows, output);

    // Synchronize and clean up allocation
    cudaDeviceSynchronize();
    cudaFree(d_window_sums);
}
