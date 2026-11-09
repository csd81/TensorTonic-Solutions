#include <cuda_runtime.h>

// CUDA kernel to perform the parallel merge using the Merge Path approach
__global__ void merge_path_kernel(const float* A, int M, const float* B, int N, float* C) {
    int total_elements = M + N;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop to handle arbitrary sizes and configuration mapping
    for (int idx = tid; idx < total_elements; idx += stride) {
        // Binary search to find the diagonal intersection (Merge Path)
        int low = max(0, idx - N);
        int high = min(idx, M);
        
        while (low < high) {
            int mid = low + ((high - low) >> 1);
            int idx_B = idx - 1 - mid;
            
            if (A[mid] <= B[idx_B]) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        
        int pos_A = low;
        int pos_B = idx - low;

        // Determine which element belongs at output index 'idx'
        if (pos_A < M && (pos_B >= N || A[pos_A] <= B[pos_B])) {
            C[idx] = A[pos_A];
        } else {
            C[idx] = B[pos_B];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    int total_elements = M + N;
    if (total_elements == 0) return;

    // Define execution configuration
    int threads_per_block = 256;
    // Cap the grid size to maximize occupancy without excessive overhead
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 65535) {
        blocks_per_grid = 65535; 
    }

    // Launch the kernel on the current default stream
    merge_path_kernel<<<blocks_per_grid, threads_per_block>>>(A, M, B, N, C);
    
    // Optional: Synchronize device if host blocking behavior is required immediately
    // cudaDeviceSynchronize();
}
