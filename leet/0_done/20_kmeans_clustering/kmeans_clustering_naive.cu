#include <cuda_runtime.h>

#include <cfloat>

// Naive k-means: a host-driven loop with one kernel launch per phase, one GPU
// thread per point for the assignment, and direct global-memory atomics for
// the per-cluster sums.  Straight assign -> update -> convergence test ->
// final assign, exactly the reference semantics, with no shared-memory
// reduction, no warp shuffles and no cooperative launch.

#define KM_THREADS 256

__global__ void km_assign_kernel(const float* __restrict__ data_x,
                                 const float* __restrict__ data_y,
                                 int* __restrict__ labels,
                                 const float* __restrict__ cx,
                                 const float* __restrict__ cy,
                                 float* __restrict__ sum_x,
                                 float* __restrict__ sum_y,
                                 int* __restrict__ count, int n, int k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float px = data_x[i];
  const float py = data_y[i];
  float best = FLT_MAX;
  int bc = 0;
  for (int c = 0; c < k; ++c) {
    const float ddx = px - cx[c];
    const float ddy = py - cy[c];
    const float d = ddx * ddx + ddy * ddy;
    if (d < best) {
      best = d;
      bc = c;
    }
  }
  labels[i] = bc;
  atomicAdd(&sum_x[bc], px);
  atomicAdd(&sum_y[bc], py);
  atomicAdd(&count[bc], 1);
}

__global__ void km_update_kernel(float* __restrict__ cx, float* __restrict__ cy,
                                 float* __restrict__ sum_x,
                                 float* __restrict__ sum_y,
                                 int* __restrict__ count, int k,
                                 int* __restrict__ moved) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;

  const int cnt = count[c];
  if (cnt > 0) {
    const float nx = sum_x[c] / static_cast<float>(cnt);
    const float ny = sum_y[c] / static_cast<float>(cnt);
    if (fabsf(nx - cx[c]) > 1e-6f || fabsf(ny - cy[c]) > 1e-6f) *moved = 1;
    cx[c] = nx;
    cy[c] = ny;
  }
  // Reset the accumulators for the next iteration.
  sum_x[c] = 0.0f;
  sum_y[c] = 0.0f;
  count[c] = 0;
}

__global__ void km_assign_final_kernel(const float* __restrict__ data_x,
                                       const float* __restrict__ data_y,
                                       int* __restrict__ labels,
                                       const float* __restrict__ cx,
                                       const float* __restrict__ cy, int n,
                                       int k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float px = data_x[i];
  const float py = data_y[i];
  float best = FLT_MAX;
  int bc = 0;
  for (int c = 0; c < k; ++c) {
    const float ddx = px - cx[c];
    const float ddy = py - cy[c];
    const float d = ddx * ddx + ddy * ddy;
    if (d < best) {
      best = d;
      bc = c;
    }
  }
  labels[i] = bc;
}

// data_x, data_y, labels, initial_centroid_x/y, final_centroid_x/y are device
// pointers.
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y,
                      float* final_centroid_x, float* final_centroid_y,
                      int sample_size, int k, int max_iterations) {
  if (sample_size <= 0 || k <= 0 || max_iterations < 0) return;

  const size_t kbytes = static_cast<size_t>(k) * sizeof(float);
  cudaMemcpyAsync(final_centroid_x, initial_centroid_x, kbytes,
                  cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(final_centroid_y, initial_centroid_y, kbytes,
                  cudaMemcpyDeviceToDevice);

  float* sum_x = nullptr;
  float* sum_y = nullptr;
  int* count = nullptr;
  int* moved = nullptr;
  cudaMalloc(&sum_x, kbytes);
  cudaMalloc(&sum_y, kbytes);
  cudaMalloc(&count, static_cast<size_t>(k) * sizeof(int));
  cudaMalloc(&moved, sizeof(int));
  cudaMemsetAsync(sum_x, 0, kbytes);
  cudaMemsetAsync(sum_y, 0, kbytes);
  cudaMemsetAsync(count, 0, static_cast<size_t>(k) * sizeof(int));
  cudaMemsetAsync(moved, 0, sizeof(int));

  const int point_blocks = (sample_size + KM_THREADS - 1) / KM_THREADS;
  const int cluster_blocks = (k + KM_THREADS - 1) / KM_THREADS;

  for (int it = 0; it < max_iterations; ++it) {
    km_assign_kernel<<<point_blocks, KM_THREADS>>>(
        data_x, data_y, labels, final_centroid_x, final_centroid_y, sum_x,
        sum_y, count, sample_size, k);
    km_update_kernel<<<cluster_blocks, KM_THREADS>>>(
        final_centroid_x, final_centroid_y, sum_x, sum_y, count, k, moved);

    int host_moved = 0;
    cudaMemcpy(&host_moved, moved, sizeof(int), cudaMemcpyDeviceToHost);
    if (!host_moved) break;
    cudaMemsetAsync(moved, 0, sizeof(int));
  }

  // Classify once more so the labels match the final centroids.
  km_assign_final_kernel<<<point_blocks, KM_THREADS>>>(
      data_x, data_y, labels, final_centroid_x, final_centroid_y, sample_size,
      k);

  cudaFree(sum_x);
  cudaFree(sum_y);
  cudaFree(count);
  cudaFree(moved);
  cudaDeviceSynchronize();
}
