#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 64
#define DIVUP(m, n) (((m) + (n) - 1) / (n))

// Device helper to compute Intersection over Union (IoU)
__device__ inline float compute_iou(const float* a, const float* b) {
    float x1 = max(a[0], b[0]);
    float y1 = max(a[1], b[1]);
    float x2 = min(a[2], b[2]);
    float y2 = min(a[3], b[3]);
    
    float width = max(0.0f, x2 - x1);
    float height = max(0.0f, y2 - y1);
    float inter_area = width * height;
    
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    
    return inter_area / (area_a + area_b - inter_area);
}

// 1. Bitmask calculation kernel
__global__ void nms_mask_kernel(const float* boxes, const int* sorted_idx, 
                                unsigned long long* mask, int N, float iou_threshold) {
    int row = blockIdx.y;
    int col = blockIdx.x;

    if (row > col) return;

    int thread_idx = threadIdx.x;
    int row_idx = row * THREADS_PER_BLOCK + thread_idx;
    int col_start_idx = col * THREADS_PER_BLOCK;

    __shared__ float shared_boxes[THREADS_PER_BLOCK * 4];

    if (thread_idx < THREADS_PER_BLOCK && (col_start_idx + thread_idx) < N) {
        int idx = sorted_idx[col_start_idx + thread_idx];
        shared_boxes[thread_idx * 4 + 0] = boxes[idx * 4 + 0];
        shared_boxes[thread_idx * 4 + 1] = boxes[idx * 4 + 1];
        shared_boxes[thread_idx * 4 + 2] = boxes[idx * 4 + 2];
        shared_boxes[thread_idx * 4 + 3] = boxes[idx * 4 + 3];
    }
    __syncthreads();

    if (row_idx < N) {
        int cur_box_idx = sorted_idx[row_idx];
        const float* cur_box = boxes + cur_box_idx * 4;
        unsigned long long t_mask = 0;
        int max_offset = min(THREADS_PER_BLOCK, N - col_start_idx);

        for (int i = 0; i < max_offset; i++) {
            if (row * THREADS_PER_BLOCK + thread_idx < col_start_idx + i) {
                if (compute_iou(cur_box, shared_boxes + i * 4) > iou_threshold) {
                    t_mask |= (1ULL << i);
                }
            }
        }
        int mask_cols = DIVUP(N, THREADS_PER_BLOCK);
        mask[row_idx * mask_cols + col] = t_mask;
    }
}

// Host bitonic sort network for indices based on scores
__global__ void bitonic_sort_step(int* sorted_idx, const float* scores, int N, int j, int k) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= N) return;

    int ixj = i ^ j;
    if (ixj > i) {
        if (ixj < N) {
            int idx1 = sorted_idx[i];
            int idx2 = sorted_idx[ixj];
            float score1 = scores[idx1];
            float score2 = scores[idx2];

            bool swap = false;
            if ((i & k) == 0) {
                if (score1 < score2 || (score1 == score2 && idx1 > idx2)) swap = true;
            } else {
                if (score1 > score2 || (score1 == score2 && idx1 < idx2)) swap = true;
            }

            if (swap) {
                sorted_idx[i] = idx2;
                sorted_idx[ixj] = idx1;
            }
        }
    }
}

// 2. Host orchestrator matching the exact requested signature
extern "C" void solve(const float* boxes, const float* scores, int* keep, int N, float iou_threshold) {
    if (N <= 0) return;

    // Allocate sorted index tracking array
    int* d_sorted_idx;
    cudaMalloc(&d_sorted_idx, N * sizeof(int));
    
    // Initialize indices sequentially on host and push to device
    int* h_sorted_idx = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) h_sorted_idx[i] = i;
    cudaMemcpy(d_sorted_idx, h_sorted_idx, N * sizeof(int), cudaMemcpyHostToDevice);

    // Parallel Bitonic Sort down to highest score -> lowest score sequence
    int next_pow2 = 1;
    while (next_pow2 < N) next_pow2 <<= 1;
    
    int threads = 512;
    int blocks = DIVUP(next_pow2, threads);

    for (int k = 2; k <= next_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<blocks, threads>>>(d_sorted_idx, scores, N, j, k);
        }
    }
    cudaDeviceSynchronize();

    // Map overlap lookup tables into 64-bit compact masks
    int mask_cols = DIVUP(N, THREADS_PER_BLOCK);
    unsigned long long* d_mask;
    cudaMalloc(&d_mask, N * mask_cols * sizeof(unsigned long long));
    cudaMemset(d_mask, 0, N * mask_cols * sizeof(unsigned long long));

    dim3 blocks_grid(mask_cols, mask_cols);
    nms_mask_kernel<<<blocks_grid, THREADS_PER_BLOCK>>>(boxes, d_sorted_idx, d_mask, N, iou_threshold);
    cudaDeviceSynchronize();

    // Retrieve computed values back to host for dead-simple evaluation chain
    cudaMemcpy(h_sorted_idx, d_sorted_idx, N * sizeof(int), cudaMemcpyDeviceToHost);
    unsigned long long* h_mask = (unsigned long long*)malloc(N * mask_cols * sizeof(unsigned long long));
    cudaMemcpy(h_mask, d_mask, N * mask_cols * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Initialize all entries inside keep to 0 state 
    int* h_keep = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_keep[i] = 0;

    // Fast linear evaluation using our bitmasks 
    unsigned long long* suppressed = (unsigned long long*)calloc(mask_cols, sizeof(unsigned long long));

    for (int i = 0; i < N; i++) {
        int block_id = i / THREADS_PER_BLOCK;
        int thread_id = i % THREADS_PER_BLOCK;

        if (!(suppressed[block_id] & (1ULL << thread_id))) {
            h_keep[h_sorted_idx[i]] = 1; // Mark surviving original index location
            unsigned long long* row_mask = h_mask + i * mask_cols;
            for (int j = block_id; j < mask_cols; j++) {
                suppressed[j] |= row_mask[j];
            }
        }
    }

    // Ship results back to target destination device memory layout
    cudaMemcpy(keep, h_keep, N * sizeof(int), cudaMemcpyHostToDevice);

    // Clean up all allocated structures cleanly
    free(h_sorted_idx);
    free(h_mask);
    free(h_keep);
    free(suppressed);
    cudaFree(d_sorted_idx);
    cudaFree(d_mask);
}
