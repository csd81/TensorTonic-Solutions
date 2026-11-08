#include <cuda_runtime.h>
#include <float.h>

// Kernel 1: Assign each 2D point to the nearest centroid
__global__ void assign_clusters_kernel(const float* __restrict__ data_x,
                                       const float* __restrict__ data_y,
                                       int* __restrict__ labels,
                                       const float* __restrict__ centroid_x,
                                       const float* __restrict__ centroid_y,
                                       int sample_size, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_size) return;

    float px = data_x[idx];
    float py = data_y[idx];

    float min_dist = FLT_MAX;
    int best_cluster = 0;

    // Compare squared Euclidean distance against all k centroids
    for (int c = 0; c < k; ++c) {
        float dx = px - centroid_x[c];
        float dy = py - centroid_y[c];
        float dist = dx * dx + dy * dy;

        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }

    labels[idx] = best_cluster;
}

// Kernel 2: Sum x/y coordinates and count points per cluster
__global__ void accumulate_centroids_kernel(const float* __restrict__ data_x,
                                            const float* __restrict__ data_y,
                                            const int* __restrict__ labels,
                                            float* __restrict__ sum_x,
                                            float* __restrict__ sum_y,
                                            int* __restrict__ counts,
                                            int sample_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_size) return;

    int cluster = labels[idx];
    atomicAdd(&sum_x[cluster], data_x[idx]);
    atomicAdd(&sum_y[cluster], data_y[idx]);
    atomicAdd(&counts[cluster], 1);
}

// Kernel 3: Compute new centroid averages on the device
__global__ void update_centroids_kernel(float* __restrict__ centroid_x,
                                        float* __restrict__ centroid_y,
                                        const float* __restrict__ sum_x,
                                        const float* __restrict__ sum_y,
                                        const int* __restrict__ counts,
                                        int* __restrict__ changed_flag,
                                        int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    int count = counts[c];
    if (count > 0) {
        float new_x = sum_x[c] / static_cast<float>(count);
        float new_y = sum_y[c] / static_cast<float>(count);

        // Check if centroid moved beyond a tiny epsilon
        if (fabsf(new_x - centroid_x[c]) > 1e-6f || fabsf(new_y - centroid_y[c]) > 1e-6f) {
            *changed_flag = 1;
        }

        centroid_x[c] = new_x;
        centroid_y[c] = new_y;
    }
}

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {
    if (sample_size <= 0 || k <= 0) return;

    // Copy initial centroids into the final centroid buffers to start
    cudaMemcpy(final_centroid_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(final_centroid_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);

    // Allocate temporary device buffers for accumulation and convergence check
    float *d_sum_x = nullptr, *d_sum_y = nullptr;
    int *d_counts = nullptr, *d_changed = nullptr;

    cudaMalloc(&d_sum_x, k * sizeof(float));
    cudaMalloc(&d_sum_y, k * sizeof(float));
    cudaMalloc(&d_counts, k * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    int threads_per_block = 256;
    int sample_blocks = (sample_size + threads_per_block - 1) / threads_per_block;
    int k_blocks = (k + threads_per_block - 1) / threads_per_block;

    for (int iter = 0; iter < max_iterations; ++iter) {
        // 1. Assign points to nearest centroids
        assign_clusters_kernel<<<sample_blocks, threads_per_block>>>(
            data_x, data_y, labels, final_centroid_x, final_centroid_y, sample_size, k
        );

        // 2. Reset accumulators
        cudaMemset(d_sum_x, 0, k * sizeof(float));
        cudaMemset(d_sum_y, 0, k * sizeof(float));
        cudaMemset(d_counts, 0, k * sizeof(int));
        cudaMemset(d_changed, 0, sizeof(int));

        // 3. Accumulate coordinates and counts per cluster
        accumulate_centroids_kernel<<<sample_blocks, threads_per_block>>>(
            data_x, data_y, labels, d_sum_x, d_sum_y, d_counts, sample_size
        );

        // 4. Update centroid positions
        update_centroids_kernel<<<k_blocks, threads_per_block>>>(
            final_centroid_x, final_centroid_y, d_sum_x, d_sum_y, d_counts, d_changed, k
        );

        // 5. Early exit if centroids have converged
        int h_changed = 0;
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_changed == 0) {
            break;
        }
    }

    // Ensure final labels match the final centroids if early exit didn't happen on the last update
    assign_clusters_kernel<<<sample_blocks, threads_per_block>>>(
        data_x, data_y, labels, final_centroid_x, final_centroid_y, sample_size, k
    );

    // Clean up temporary allocations
    cudaFree(d_sum_x);
    cudaFree(d_sum_y);
    cudaFree(d_counts);
    cudaFree(d_changed);
}
