// kmeans.cu -- CUDA C++ port of kmeans.py (Numba CUDA implementation).
//
// Build (Colab T4 / CUDA 12+):
//   nvcc -O3 -arch=native -o kmeans kmeans.cu
// Alternative arch flags: -arch=sm_75 (T4), -arch=sm_61 (Quadro P2000).
//
// Notes:
//  * Input is a flat row-major float32 array of shape (n_points, dim).
//  * Initialization samples k distinct points using std::mt19937; this does
//    NOT reproduce NumPy's RandomState sequence, so centroids will differ
//    from the Python version even for the same seed.
//  * The per-phase timings mirror run_gpu_kmeans() in kmeans.py.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                    \
                   cudaGetErrorName(err__), __FILE__, __LINE__,               \
                   cudaGetErrorString(err__));                                \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static double elapsed_s(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void kmeans_assign_kernel(const float* __restrict__ X,
                                     const float* __restrict__ centroids,
                                     int* __restrict__ labels, int n_points,
                                     int k, int dim) {
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int idx = start; idx < n_points; idx += stride) {
    const float* point = X + static_cast<size_t>(idx) * dim;
    float min_dist = 1e30f;
    int best_cluster = 0;

    for (int j = 0; j < k; ++j) {
      const float* c = centroids + static_cast<size_t>(j) * dim;
      float dist_sq = 0.0f;
      for (int d = 0; d < dim; ++d) {
        const float diff = point[d] - c[d];
        dist_sq = fmaf(diff, diff, dist_sq);
      }
      if (dist_sq < min_dist) {
        min_dist = dist_sq;
        best_cluster = j;
      }
    }
    labels[idx] = best_cluster;
  }
}

__global__ void kmeans_update_kernel(const float* __restrict__ X,
                                     const int* __restrict__ labels,
                                     float* __restrict__ new_centroids,
                                     int* __restrict__ cluster_counts,
                                     int n_points, int dim) {
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = start; i < n_points; i += stride) {
    const int c_idx = labels[i];
    const float* point = X + static_cast<size_t>(i) * dim;
    float* sum = new_centroids + static_cast<size_t>(c_idx) * dim;

    for (int d = 0; d < dim; ++d) {
      atomicAdd(sum + d, point[d]);
    }
    atomicAdd(cluster_counts + c_idx, 1);
  }
}

__global__ void zero_buffers_kernel(float* new_centroids, int* cluster_counts,
                                    int total, int k) {
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = start; i < total; i += stride) {
    new_centroids[i] = 0.0f;
  }
  for (int i = start; i < k; i += stride) {
    cluster_counts[i] = 0;
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct KMeansStats {
  double setup_ms = 0.0;
  double assign_ms = 0.0;
  double zero_ms = 0.0;
  double update_ms = 0.0;
  double d2h_ms = 0.0;
  double h2d_ms = 0.0;
  double host_ms = 0.0;
  double total_s = 0.0;
  int iterations = 0;
  bool converged = false;
};

static void validate_inputs(int n_points, int dim, int k, int max_iters) {
  if (n_points < 1) {
    throw std::invalid_argument("X must contain at least one point");
  }
  if (dim < 1) {
    throw std::invalid_argument("X must have at least one feature");
  }
  if (k < 1 || k > n_points) {
    throw std::invalid_argument("k must satisfy 1 <= k <= n_points");
  }
  if (max_iters < 1) {
    throw std::invalid_argument("max_iters must be >= 1");
  }
}

// h_X: row-major float32, n_points x dim. Returns centroids (k x dim) and
// labels (n_points) through the output vectors, plus per-phase timings.
static KMeansStats run_gpu_kmeans(const float* h_X, int n_points, int dim,
                                  int k, int max_iters, unsigned seed,
                                  std::vector<float>& centroids_out,
                                  std::vector<int>& labels_out) {
  validate_inputs(n_points, dim, k, max_iters);

  // Deterministic initialization: pick k distinct indices (partial
  // Fisher-Yates), then copy those points as the initial centroids.
  std::mt19937 rng(seed);
  std::vector<int> indices(n_points);
  std::iota(indices.begin(), indices.end(), 0);
  for (int i = 0; i < k; ++i) {
    std::uniform_int_distribution<int> pick(i, n_points - 1);
    const int j = pick(rng);
    std::swap(indices[i], indices[j]);
  }

  std::vector<float> centroids(static_cast<size_t>(k) * dim);
  for (int j = 0; j < k; ++j) {
    std::memcpy(centroids.data() + static_cast<size_t>(j) * dim,
                h_X + static_cast<size_t>(indices[j]) * dim,
                sizeof(float) * dim);
  }

  KMeansStats stats;

  float* d_X = nullptr;
  float* d_centroids = nullptr;
  float* d_new_centroids = nullptr;
  int* d_labels = nullptr;
  int* d_cluster_counts = nullptr;

  const size_t x_bytes = static_cast<size_t>(n_points) * dim * sizeof(float);
  const size_t centroid_bytes = static_cast<size_t>(k) * dim * sizeof(float);

  const auto setup_t0 = Clock::now();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_X), x_bytes));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_centroids), centroid_bytes));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_new_centroids), centroid_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_labels),
                        static_cast<size_t>(n_points) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cluster_counts),
                        static_cast<size_t>(k) * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_X, h_X, x_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(), centroid_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_labels, 0, static_cast<size_t>(n_points) * sizeof(int)));
  CUDA_CHECK(cudaDeviceSynchronize());
  stats.setup_ms = elapsed_ms(setup_t0);

  const int threads = THREADS_PER_BLOCK;
  const int blocks = static_cast<int>(std::min<long long>(
      65535LL, (static_cast<long long>(n_points) + threads - 1) / threads));
  const int zero_total = k * dim;
  const int zero_blocks = static_cast<int>(std::min<long long>(
      65535LL, (static_cast<long long>(std::max(zero_total, k)) + threads - 1) /
                  threads));

  cudaEvent_t a0, a1, z0, z1, u0, u1;
  CUDA_CHECK(cudaEventCreate(&a0));
  CUDA_CHECK(cudaEventCreate(&a1));
  CUDA_CHECK(cudaEventCreate(&z0));
  CUDA_CHECK(cudaEventCreate(&z1));
  CUDA_CHECK(cudaEventCreate(&u0));
  CUDA_CHECK(cudaEventCreate(&u1));

  std::vector<float> h_new_centroids(static_cast<size_t>(k) * dim);
  std::vector<int> h_counts(k);
  std::vector<float> new_centroids(static_cast<size_t>(k) * dim);

  const auto total_t0 = Clock::now();

  for (int it = 0; it < max_iters; ++it) {
    stats.iterations = it + 1;

    // 1. Assignment phase
    CUDA_CHECK(cudaEventRecord(a0));
    kmeans_assign_kernel<<<blocks, threads>>>(d_X, d_centroids, d_labels,
                                              n_points, k, dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(a1));
    CUDA_CHECK(cudaDeviceSynchronize());
    float phase_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&phase_ms, a0, a1));
    stats.assign_ms += phase_ms;

    // 2. Zero scratch buffers, then accumulate sums and counts
    CUDA_CHECK(cudaEventRecord(z0));
    zero_buffers_kernel<<<zero_blocks, threads>>>(d_new_centroids,
                                                  d_cluster_counts, zero_total,
                                                  k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(z1));
    CUDA_CHECK(cudaEventRecord(u0));
    kmeans_update_kernel<<<blocks, threads>>>(d_X, d_labels, d_new_centroids,
                                              d_cluster_counts, n_points, dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(u1));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventElapsedTime(&phase_ms, z0, z1));
    stats.zero_ms += phase_ms;
    CUDA_CHECK(cudaEventElapsedTime(&phase_ms, u0, u1));
    stats.update_ms += phase_ms;

    // 3. Device-to-host: block sums and counts
    const auto d2h_t0 = Clock::now();
    CUDA_CHECK(cudaMemcpy(h_new_centroids.data(), d_new_centroids,
                          centroid_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_cluster_counts,
                          static_cast<size_t>(k) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    stats.d2h_ms += elapsed_ms(d2h_t0);

    // 4. Host-side averaging (empty clusters keep their previous centroid)
    const auto host_t0 = Clock::now();
    for (int j = 0; j < k; ++j) {
      if (h_counts[j] > 0) {
        const float inv_count = 1.0f / static_cast<float>(h_counts[j]);
        for (int d = 0; d < dim; ++d) {
          const size_t idx = static_cast<size_t>(j) * dim + d;
          new_centroids[idx] = h_new_centroids[idx] * inv_count;
        }
      } else {
        for (int d = 0; d < dim; ++d) {
          const size_t idx = static_cast<size_t>(j) * dim + d;
          new_centroids[idx] = centroids[idx];
        }
      }
    }
    stats.host_ms += elapsed_ms(host_t0);

    // 5. Convergence check (same tolerance as the Numba version)
    bool close = true;
    for (int i = 0; i < k * dim; ++i) {
      const float diff = std::fabs(new_centroids[i] - centroids[i]);
      if (diff > 1e-4f + 1e-5f * std::fabs(centroids[i])) {
        close = false;
        break;
      }
    }
    centroids = new_centroids;
    if (close) {
      stats.converged = true;
      break;
    }

    const auto h2d_t0 = Clock::now();
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(), centroid_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.h2d_ms += elapsed_ms(h2d_t0);
  }

  // Final assignment pass so returned labels match returned centroids.
  CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(), centroid_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(a0));
  kmeans_assign_kernel<<<blocks, threads>>>(d_X, d_centroids, d_labels,
                                            n_points, k, dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(a1));
  CUDA_CHECK(cudaDeviceSynchronize());
  float final_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&final_ms, a0, a1));
  stats.assign_ms += final_ms;

  labels_out.resize(n_points);
  CUDA_CHECK(cudaMemcpy(labels_out.data(), d_labels,
                        static_cast<size_t>(n_points) * sizeof(int),
                        cudaMemcpyDeviceToHost));

  centroids_out = centroids;
  stats.total_s = elapsed_s(total_t0);

  CUDA_CHECK(cudaEventDestroy(a0));
  CUDA_CHECK(cudaEventDestroy(a1));
  CUDA_CHECK(cudaEventDestroy(z0));
  CUDA_CHECK(cudaEventDestroy(z1));
  CUDA_CHECK(cudaEventDestroy(u0));
  CUDA_CHECK(cudaEventDestroy(u1));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_centroids));
  CUDA_CHECK(cudaFree(d_new_centroids));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_cluster_counts));

  return stats;
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\n", prop.name);

  constexpr int N = 200000;
  constexpr int D = 8;
  constexpr int K = 8;
  constexpr int MAX_ITERS = 15;

  std::mt19937 data_gen(42);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> X(static_cast<size_t>(N) * D);
  for (float& value : X) {
    value = normal(data_gen);
  }

  std::printf("N=%d, D=%d, K=%d. Running GPU k-means...\n", N, D, K);

  std::vector<float> centroids;
  std::vector<int> labels;
  KMeansStats stats =
      run_gpu_kmeans(X.data(), N, D, K, MAX_ITERS, 42, centroids, labels);

  std::printf("Done. GPU total time: %.4f s\n", stats.total_s);
  std::printf("  assign: %.3f ms | zero: %.3f ms | update: %.3f ms\n",
              stats.assign_ms, stats.zero_ms, stats.update_ms);
  std::printf("  d2h: %.3f ms | h2d: %.3f ms | host: %.3f ms\n",
              stats.d2h_ms, stats.h2d_ms, stats.host_ms);
  std::printf("  setup: %.3f ms | iterations: %d | converged: %s\n",
              stats.setup_ms, stats.iterations,
              stats.converged ? "true" : "false");

  return EXIT_SUCCESS;
}
