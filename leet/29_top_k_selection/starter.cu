#include <cuda_runtime.h>
#include <float.h>
#include <algorithm>

#define BLOCK_SIZE 256
#define MAX_FAST_K 256

// Helper: Insert value into a thread-local min-heap of size k
__device__ __forceinline__ void heap_insert(float* heap, int k, float val) {
    if (val <= heap[0]) return;
    heap[0] = val;
    int idx = 0;
    while (true) {
        int left = 2 * idx + 1;
        int right = left + 1;
        int smallest = idx;
        if (left < k && heap[left] < heap[smallest]) smallest = left;
        if (right < k && heap[right] < heap[smallest]) smallest = right;
        if (smallest == idx) break;
        float tmp = heap[idx];
        heap[idx] = heap[smallest];
        heap[smallest] = tmp;
        idx = smallest;
    }
}

// Kernel for small/medium k (k <= MAX_FAST_K)
// Each block reduces a portion of `input` into `block_out` of size k (sorted descending)
__global__ void topk_stage1_kernel(const float* __restrict__ input, float* __restrict__ block_out, int N, int k) {
    extern __shared__ float smem[]; // Size: BLOCK_SIZE * k

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // 1. Thread-local min-heap in registers/local memory
    float local_heap[MAX_FAST_K];
    for (int i = 0; i < k; ++i) {
        local_heap[i] = -FLT_MAX;
    }

    // Grid-stride loop to process multiple elements per thread
    for (int i = gid; i < N; i += stride) {
        heap_insert(local_heap, k, input[i]);
    }

    // Write thread-local heap to shared memory
    for (int i = 0; i < k; ++i) {
        smem[tid * k + i] = local_heap[i];
    }
    __syncthreads();

    // 2. Parallel tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float* my_heap = &smem[tid * k];
            float* other_heap = &smem[(tid + s) * k];

            // Heapify my_heap just in case, then insert elements from other_heap
            // Since local_heap is already a valid min-heap, we can directly insert
            for (int i = 0; i < k; ++i) {
                heap_insert(my_heap, k, other_heap[i]);
            }
        }
        __syncthreads();
    }

    // 3. Thread 0 sorts the block's top-k min-heap into descending order
    if (tid == 0) {
        float* final_heap = &smem[0];
        // Heap sort: repeatedly extract min and place at the end
        for (int end = k - 1; end > 0; --end) {
            float min_val = final_heap[0];
            final_heap[0] = final_heap[end];
            final_heap[end] = min_val;

            // Sift down in heap of size `end`
            int idx = 0;
            while (true) {
                int left = 2 * idx + 1;
                int right = left + 1;
                int smallest = idx;
                if (left < end && final_heap[left] < final_heap[smallest]) smallest = left;
                if (right < end && final_heap[right] < final_heap[smallest]) smallest = right;
                if (smallest == idx) break;
                float tmp = final_heap[idx];
                final_heap[idx] = final_heap[smallest];
                final_heap[smallest] = tmp;
                idx = smallest;
            }
        }

        // Write sorted block result to global memory
        for (int i = 0; i < k; ++i) {
            block_out[blockIdx.x * k + i] = final_heap[i];
        }
    }
}

// Fallback Kernel: Parallel Bitonic Sort step for large k (k > MAX_FAST_K)
// Maps float bits to sortable unsigned int so negative floats order properly
__device__ __forceinline__ unsigned int float_to_ordered_uint(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000) ? ~u : (u | 0x80000000);
}

__device__ __forceinline__ float ordered_uint_to_float(unsigned int u) {
    u = (u & 0x80000000) ? (u & ~0x80000000) : ~u;
    return __uint_as_float(u);
}

__global__ void pad_and_convert_kernel(const float* __restrict__ in, unsigned int* __restrict__ out, int N, int padded_N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = float_to_ordered_uint(in[idx]);
    } else if (idx < padded_N) {
        out[idx] = 0; // Smallest possible sortable uint (-NaN / -Inf)
    }
}

__global__ void bitonic_sort_step(unsigned int* data, int j, int k_step) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    if (ixj > i) {
        bool descending = ((i & k_step) == 0);
        unsigned int a = data[i];
        unsigned int b = data[ixj];
        if ((descending && a < b) || (!descending && a > b)) {
            data[i] = b;
            data[ixj] = a;
        }
    }
}

__global__ void extract_top_k_kernel(const unsigned int* __restrict__ sorted, float* __restrict__ output, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        output[idx] = ordered_uint_to_float(sorted[idx]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    if (N <= 0 || k <= 0) return;

    // Fast Path: Two-stage Min-Heap Reduction (Ideal for k <= 256, including N=50M, k=100)
    if (k <= MAX_FAST_K) {
        int num_blocks = std::min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 512);
        // Ensure num_blocks is a power of 2 or at least efficient for Stage 2
        if (num_blocks < 1) num_blocks = 1;

        size_t smem_bytes = BLOCK_SIZE * k * sizeof(float);

        if (num_blocks == 1) {
            topk_stage1_kernel<<<1, BLOCK_SIZE, smem_bytes>>>(input, output, N, k);
        } else {
            float* d_temp = nullptr;
            cudaMalloc(&d_temp, num_blocks * k * sizeof(float));

            // Stage 1: Reduce N elements down to (num_blocks * k) candidates
            topk_stage1_kernel<<<num_blocks, BLOCK_SIZE, smem_bytes>>>(input, d_temp, N, k);

            // Stage 2: Reduce (num_blocks * k) candidates down to final k elements
            topk_stage1_kernel<<<1, BLOCK_SIZE, smem_bytes>>>(d_temp, output, num_blocks * k, k);

            cudaFree(d_temp);
        }
    } 
    // Fallback Path: Full Bitonic Sort for large k (k > 256)
    else {
        int padded_N = 1;
        while (padded_N < N) padded_N <<= 1;

        unsigned int* d_data = nullptr;
        cudaMalloc(&d_data, padded_N * sizeof(unsigned int));

        int blocks = (padded_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_and_convert_kernel<<<blocks, BLOCK_SIZE>>>(input, d_data, N, padded_N);

        for (int k_step = 2; k_step <= padded_N; k_step <<= 1) {
            for (int j = k_step >> 1; j > 0; j >>= 1) {
                bitonic_sort_step<<<blocks, BLOCK_SIZE>>>(d_data, j, k_step);
            }
        }

        int out_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        extract_top_k_kernel<<<out_blocks, BLOCK_SIZE>>>(d_data, output, k);

        cudaFree(d_data);
    }
}
