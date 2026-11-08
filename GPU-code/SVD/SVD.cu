// SVD.cu -- one-sided Jacobi SVD in CUDA C++.
//
// Port of SVD.py:
//   * A is stored column-major as At (n_work x m); row j is column j of A.
//   * For odd n, one zero column is appended (n_work = n + 1).
//   * Each sweep runs the round-robin ("circle method") schedule of
//     n_work - 1 sub-sweeps with n_work / 2 disjoint column pairs.
//   * jacobi_dots_kernel computes alpha, beta, gamma per pair with a
//     shared-memory reduction; jacobi_rotate_kernel computes cos/sin and
//     applies the Givens rotation to At and to Vt (accumulated rotations).
//   * Column norms are the singular values, normalized columns are U, and
//     Vt (after sorting rows by the singular-value order) is V^T.
//
// Requires m >= n.  Build:
//   nvcc -O3 -arch=native -o svd SVD.cu
// (T4: -arch=sm_75, Quadro P2000: -arch=sm_61)

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#define THREADS 256

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

__global__ void jacobi_dots_kernel(const double* __restrict__ At,
                                   const int* __restrict__ pair_p,
                                   const int* __restrict__ pair_q,
                                   double* __restrict__ alpha,
                                   double* __restrict__ beta,
                                   double* __restrict__ gamma, int round_idx,
                                   int n_pairs, int m) {
  const int pair = blockIdx.x;
  if (pair >= n_pairs) {
    return;
  }

  const int p = pair_p[round_idx * n_pairs + pair];
  const int q = pair_q[round_idx * n_pairs + pair];
  const int tid = threadIdx.x;

  double local_a = 0.0;
  double local_b = 0.0;
  double local_g = 0.0;
  for (int i = tid; i < m; i += blockDim.x) {
    const double ap = At[static_cast<size_t>(p) * m + i];
    const double aq = At[static_cast<size_t>(q) * m + i];
    local_a += ap * ap;
    local_b += aq * aq;
    local_g += ap * aq;
  }

  __shared__ double s_a[THREADS];
  __shared__ double s_b[THREADS];
  __shared__ double s_g[THREADS];
  s_a[tid] = local_a;
  s_b[tid] = local_b;
  s_g[tid] = local_g;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_a[tid] += s_a[tid + stride];
      s_b[tid] += s_b[tid + stride];
      s_g[tid] += s_g[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    alpha[pair] = s_a[0];
    beta[pair] = s_b[0];
    gamma[pair] = s_g[0];
  }
}

__global__ void jacobi_rotate_kernel(double* __restrict__ At,
                                     double* __restrict__ Vt,
                                     const int* __restrict__ pair_p,
                                     const int* __restrict__ pair_q,
                                     const double* __restrict__ alpha,
                                     const double* __restrict__ beta,
                                     const double* __restrict__ gamma,
                                     int* __restrict__ rot_count, int round_idx,
                                     int n_pairs, int m, int n_work,
                                     double tol) {
  const int pair = blockIdx.x;
  if (pair >= n_pairs) {
    return;
  }

  const int p = pair_p[round_idx * n_pairs + pair];
  const int q = pair_q[round_idx * n_pairs + pair];
  const int tid = threadIdx.x;

  __shared__ double s_c;
  __shared__ double s_s;
  __shared__ int s_do_rotate;

  if (tid == 0) {
    const double a = alpha[pair];
    const double b = beta[pair];
    const double g = gamma[pair];
    const double scale = sqrt(a) * sqrt(b);

    if (scale == 0.0 || fabs(g) <= tol * scale) {
      s_c = 1.0;
      s_s = 0.0;
      s_do_rotate = 0;
    } else {
      const double zeta = (b - a) / (2.0 * g);
      const double sign = (zeta >= 0.0) ? 1.0 : -1.0;
      const double t = sign / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
      s_c = 1.0 / sqrt(1.0 + t * t);
      s_s = t * s_c;
      s_do_rotate = 1;
      atomicAdd(rot_count, 1);
    }
  }
  __syncthreads();

  if (s_do_rotate != 0) {
    const double c = s_c;
    const double s = s_s;

    for (int i = tid; i < m; i += blockDim.x) {
      const double ap = At[static_cast<size_t>(p) * m + i];
      const double aq = At[static_cast<size_t>(q) * m + i];
      At[static_cast<size_t>(p) * m + i] = c * ap - s * aq;
      At[static_cast<size_t>(q) * m + i] = s * ap + c * aq;
    }

    for (int i = tid; i < n_work; i += blockDim.x) {
      const double vp = Vt[static_cast<size_t>(p) * n_work + i];
      const double vq = Vt[static_cast<size_t>(q) * n_work + i];
      Vt[static_cast<size_t>(p) * n_work + i] = c * vp - s * vq;
      Vt[static_cast<size_t>(q) * n_work + i] = s * vp + c * vq;
    }
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct SVDResult {
  std::vector<double> U;   // m x n, row-major (thin SVD)
  std::vector<double> S;   // n
  std::vector<double> Vh;  // n x n, row-major
};

struct SVDStats {
  int sweeps = 0;
  bool converged = false;
  int n_work = 0;
  int n_pairs = 0;
  int n_rounds = 0;
  double h2d_ms = 0.0;
  double compute_ms = 0.0;
  double d2h_ms = 0.0;
  double total_s = 0.0;
};

static void make_pair_schedule(int n_work, std::vector<int>& pair_p,
                               std::vector<int>& pair_q) {
  const int n_half = n_work / 2;
  const int n_rounds = n_work - 1;
  pair_p.resize(static_cast<size_t>(n_rounds) * n_half);
  pair_q.resize(static_cast<size_t>(n_rounds) * n_half);

  for (int r = 0; r < n_rounds; ++r) {
    pair_p[static_cast<size_t>(r) * n_half + 0] = 0;
    pair_q[static_cast<size_t>(r) * n_half + 0] = r + 1;
    for (int k = 1; k < n_half; ++k) {
      // C++ % keeps the sign of the dividend, unlike Python's modulo.
      const int a = (r + k) % n_rounds;
      const int b = ((r - k) % n_rounds + n_rounds) % n_rounds;
      pair_p[static_cast<size_t>(r) * n_half + k] = a + 1;
      pair_q[static_cast<size_t>(r) * n_half + k] = b + 1;
    }
  }
}

// A: row-major m x n. tol: relative |gamma| threshold.
static SVDResult jacobi_svd(const std::vector<double>& A, int m, int n,
                            double tol = 1e-12, int max_sweeps = 30,
                            SVDStats* stats = nullptr) {
  if (m < n) {
    throw std::invalid_argument("m >= n is required");
  }
  if (n < 1) {
    throw std::invalid_argument("A must have at least one column");
  }
  if (static_cast<size_t>(m) * n != A.size()) {
    throw std::invalid_argument("A size does not match m * n");
  }

  const int n_work = n + (n & 1);
  const int n_half = n_work / 2;
  const int n_rounds = n_work - 1;

  // At[j * m + i] = A[i * n + j]; padded column (if any) stays zero.
  std::vector<double> At(static_cast<size_t>(n_work) * m, 0.0);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      At[static_cast<size_t>(j) * m + i] = A[static_cast<size_t>(i) * n + j];
    }
  }

  std::vector<double> Vt(static_cast<size_t>(n_work) * n_work, 0.0);
  for (int j = 0; j < n_work; ++j) {
    Vt[static_cast<size_t>(j) * n_work + j] = 1.0;
  }

  std::vector<int> pair_p, pair_q;
  make_pair_schedule(n_work, pair_p, pair_q);

  double* d_At = nullptr;
  double* d_Vt = nullptr;
  int* d_pair_p = nullptr;
  int* d_pair_q = nullptr;
  double* d_alpha = nullptr;
  double* d_beta = nullptr;
  double* d_gamma = nullptr;
  int* d_rot_count = nullptr;

  SVDStats local_stats;
  local_stats.n_work = n_work;
  local_stats.n_pairs = n_half;
  local_stats.n_rounds = n_rounds;

  const auto total_t0 = Clock::now();
  const auto setup_t0 = total_t0;
  const size_t at_bytes = static_cast<size_t>(n_work) * m * sizeof(double);
  const size_t vt_bytes =
      static_cast<size_t>(n_work) * n_work * sizeof(double);
  const size_t pair_bytes =
      static_cast<size_t>(n_rounds) * n_half * sizeof(int);

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_At), at_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Vt), vt_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pair_p), pair_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pair_q), pair_bytes));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_alpha), n_half * sizeof(double)));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_beta), n_half * sizeof(double)));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_gamma), n_half * sizeof(double)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rot_count), sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_At, At.data(), at_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Vt, Vt.data(), vt_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pair_p, pair_p.data(), pair_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pair_q, pair_q.data(), pair_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  local_stats.h2d_ms = elapsed_ms(setup_t0);

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  int h_rot_count = 0;
  bool converged = false;
  int sweeps = 0;

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    CUDA_CHECK(cudaMemsetAsync(d_rot_count, 0, sizeof(int)));

    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < n_rounds; ++r) {
      jacobi_dots_kernel<<<n_half, THREADS>>>(d_At, d_pair_p, d_pair_q,
                                              d_alpha, d_beta, d_gamma, r,
                                              n_half, m);
      CUDA_CHECK(cudaGetLastError());
      jacobi_rotate_kernel<<<n_half, THREADS>>>(
          d_At, d_Vt, d_pair_p, d_pair_q, d_alpha, d_beta, d_gamma,
          d_rot_count, r, n_half, m, n_work, tol);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaMemcpy(&h_rot_count, d_rot_count, sizeof(int),
                          cudaMemcpyDeviceToHost));

    float sweep_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&sweep_ms, e0, e1));
    local_stats.compute_ms += sweep_ms;

    ++sweeps;
    if (h_rot_count == 0) {
      converged = true;
      break;
    }
  }
  local_stats.sweeps = sweeps;
  local_stats.converged = converged;

  const auto d2h_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(At.data(), d_At, at_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Vt.data(), d_Vt, vt_bytes, cudaMemcpyDeviceToHost));
  local_stats.d2h_ms = elapsed_ms(d2h_t0);
  local_stats.total_s = elapsed_s(total_t0);

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaFree(d_At));
  CUDA_CHECK(cudaFree(d_Vt));
  CUDA_CHECK(cudaFree(d_pair_p));
  CUDA_CHECK(cudaFree(d_pair_q));
  CUDA_CHECK(cudaFree(d_alpha));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_rot_count));

  // Final normalization and sorting on the host.
  std::vector<double> norms(n);
  for (int j = 0; j < n; ++j) {
    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
      const double v = At[static_cast<size_t>(j) * m + i];
      sum += v * v;
    }
    norms[j] = std::sqrt(sum);
  }

  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&norms](int a, int b) { return norms[a] > norms[b]; });

  SVDResult result;
  result.S.resize(n);
  result.U.assign(static_cast<size_t>(m) * n, 0.0);
  result.Vh.assign(static_cast<size_t>(n) * n, 0.0);

  for (int j = 0; j < n; ++j) {
    const int src = order[j];
    const double sigma = norms[src];
    result.S[j] = sigma;
    if (sigma > 0.0) {
      for (int i = 0; i < m; ++i) {
        result.U[static_cast<size_t>(i) * n + j] =
            At[static_cast<size_t>(src) * m + i] / sigma;
      }
    }
    for (int i = 0; i < n; ++i) {
      result.Vh[static_cast<size_t>(j) * n + i] =
          Vt[static_cast<size_t>(src) * n_work + i];
    }
  }

  if (stats != nullptr) {
    *stats = local_stats;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\n", prop.name);

  constexpr int M = 128;
  constexpr int N = 64;

  std::mt19937 gen(0);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<double> A(static_cast<size_t>(M) * N);
  for (double& value : A) {
    value = normal(gen);
  }

  SVDStats stats;
  const SVDResult result = jacobi_svd(A, M, N, 1e-12, 30, &stats);

  // Reconstruction: C = U diag(S) Vh
  std::vector<double> C(static_cast<size_t>(M) * N, 0.0);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {
        sum += result.U[static_cast<size_t>(i) * N + k] * result.S[k] *
               result.Vh[static_cast<size_t>(k) * N + j];
      }
      C[static_cast<size_t>(i) * N + j] = sum;
    }
  }

  double diff_norm = 0.0;
  double a_norm = 0.0;
  for (size_t i = 0; i < A.size(); ++i) {
    diff_norm += (A[i] - C[i]) * (A[i] - C[i]);
    a_norm += A[i] * A[i];
  }
  const double rec_err = std::sqrt(diff_norm / a_norm);

  // Orthogonality errors
  double utu_err = 0.0;
  for (int a = 0; a < N; ++a) {
    for (int b = 0; b < N; ++b) {
      double dot = 0.0;
      for (int i = 0; i < M; ++i) {
        dot += result.U[static_cast<size_t>(i) * N + a] *
               result.U[static_cast<size_t>(i) * N + b];
      }
      const double target = (a == b) ? 1.0 : 0.0;
      utu_err = std::max(utu_err, std::fabs(dot - target));
    }
  }

  double vvt_err = 0.0;
  for (int a = 0; a < N; ++a) {
    for (int b = 0; b < N; ++b) {
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += result.Vh[static_cast<size_t>(a) * N + k] *
               result.Vh[static_cast<size_t>(b) * N + k];
      }
      const double target = (a == b) ? 1.0 : 0.0;
      vvt_err = std::max(vvt_err, std::fabs(dot - target));
    }
  }

  std::printf("matrix %dx%d, sweeps=%d, converged=%s\n", M, N, stats.sweeps,
              stats.converged ? "true" : "false");
  std::printf("time: %.3f ms (h2d %.3f + compute %.3f + d2h %.3f)\n",
              stats.total_s * 1000.0, stats.h2d_ms, stats.compute_ms,
              stats.d2h_ms);
  std::printf("reconstruction error: %.3e\n", rec_err);
  std::printf("||U^T U - I||_inf: %.3e | ||Vh Vh^T - I||_inf: %.3e\n",
              utu_err, vvt_err);
  std::printf("sigma[0..3]: %.6f %.6f %.6f %.6f\n", result.S[0], result.S[1],
              result.S[2], result.S[3]);

  return EXIT_SUCCESS;
}
