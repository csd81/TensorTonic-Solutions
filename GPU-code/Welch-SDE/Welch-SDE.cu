// Welch-SDE.cu -- Welch power spectral density estimation in CUDA C++.
//
// Port of Welch-SDE.py: a custom radix-2 Cooley-Tukey FFT kernel
// (one block per segment, re/im planes in dynamic shared memory) followed
// by a per-frequency PSD averaging kernel. No cuFFT dependency.
//
// Build:
//   nvcc -O3 -arch=native -o welch Welch-SDE.cu
// (T4: -arch=sm_75, Quadro P2000: -arch=sm_61)
//
// Constraints, matching the Python version:
//   * nperseg is a power of two, 2 <= nperseg <= 1024 (one block per segment)
//   * signal is 1-D float32
//   * complete segments only; n_seg = 1 + (N - nperseg) / step
//
// PSD normalization (two-sided):  |X_i[k]|^2 / (n_seg * sum(w^2))
// With fs != 1 the density is divided by fs; the one-sided output doubles
// every bin except DC and Nyquist.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define THREADS_PSD 256
#define MAX_NPERSEG 1024

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
static constexpr double PI = 3.14159265358979323846;

static double elapsed_ms(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static double elapsed_s(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void welch_fft_kernel(const float* __restrict__ signal,
                                 const float* __restrict__ window,
                                 const int* __restrict__ bitrev,
                                 const float* __restrict__ tw_cos,
                                 const float* __restrict__ tw_sin,
                                 float* __restrict__ out_re,
                                 float* __restrict__ out_im, int n_seg, int L,
                                 int D) {
  extern __shared__ float smem[];
  float* re = smem;
  float* im = smem + L;

  const int seg = blockIdx.x;
  const int tid = threadIdx.x;

  if (seg < n_seg && tid < L) {
    re[tid] = signal[static_cast<size_t>(seg) * D + tid] * window[tid];
    im[tid] = 0.0f;
  }
  __syncthreads();

  // Bit-reversal permutation
  if (tid < L) {
    const int j = bitrev[tid];
    if (j > tid) {
      const float tr = re[tid];
      const float ti = im[tid];
      re[tid] = re[j];
      im[tid] = im[j];
      re[j] = tr;
      im[j] = ti;
    }
  }
  __syncthreads();

  // Radix-2 butterfly stages
  for (int length = 2; length <= L; length <<= 1) {
    const int half = length >> 1;
    if (tid < (L >> 1)) {
      const int group = tid / half;
      const int offset = tid % half;
      const int i0 = group * length + offset;
      const int i1 = i0 + half;
      const int m = offset * (L / length);
      const float wr = tw_cos[m];
      const float wi = tw_sin[m];

      const float br = re[i1];
      const float bi = im[i1];
      const float tr = wr * br - wi * bi;
      const float ti = wr * bi + wi * br;

      re[i1] = re[i0] - tr;
      im[i1] = im[i0] - ti;
      re[i0] += tr;
      im[i0] += ti;
    }
    __syncthreads();
  }

  if (seg < n_seg && tid < L) {
    out_re[static_cast<size_t>(seg) * L + tid] = re[tid];
    out_im[static_cast<size_t>(seg) * L + tid] = im[tid];
  }
}

__global__ void welch_psd_kernel(const float* __restrict__ re,
                                 const float* __restrict__ im,
                                 float* __restrict__ psd, int n_seg,
                                 int n_freq, int spec_len, float scale,
                                 int onesided, int nyquist_idx) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_freq) {
    return;
  }

  float total = 0.0f;
  for (int i = 0; i < n_seg; ++i) {
    const size_t idx = static_cast<size_t>(i) * spec_len + k;
    const float a = re[idx];
    const float b = im[idx];
    total += a * a + b * b;
  }

  float factor = scale;
  if (onesided && k > 0 && k != nyquist_idx) {
    factor *= 2.0f;
  }
  psd[k] = total * factor;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static std::vector<float> make_window(const std::string& name, int L) {
  std::vector<float> w(L);
  for (int n = 0; n < L; ++n) {
    const double angle = 2.0 * PI * n / L;
    if (name == "hann") {
      w[n] = static_cast<float>(0.5 - 0.5 * std::cos(angle));
    } else if (name == "hamming") {
      w[n] = static_cast<float>(0.54 - 0.46 * std::cos(angle));
    } else if (name == "rect" || name == "boxcar") {
      w[n] = 1.0f;
    } else {
      throw std::invalid_argument(
          "unknown window; use 'hann', 'hamming' or 'rect'");
    }
  }
  return w;
}

static std::vector<int> bit_reverse_table(int L) {
  int bits = 0;
  for (int v = L; v > 1; v >>= 1) {
    ++bits;
  }
  std::vector<int> rev(L);
  for (int i = 0; i < L; ++i) {
    int value = i;
    int result = 0;
    for (int b = 0; b < bits; ++b) {
      result = (result << 1) | (value & 1);
      value >>= 1;
    }
    rev[i] = result;
  }
  return rev;
}

static void make_twiddles(int L, std::vector<float>& tw_cos,
                          std::vector<float>& tw_sin) {
  tw_cos.resize(L / 2);
  tw_sin.resize(L / 2);
  for (int m = 0; m < L / 2; ++m) {
    const double angle = 2.0 * PI * m / L;
    tw_cos[m] = static_cast<float>(std::cos(angle));
    tw_sin[m] = static_cast<float>(-std::sin(angle));
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

struct WelchResult {
  std::vector<float> freqs;
  std::vector<float> psd;
  int n_seg = 0;
  int n_freq = 0;
};

struct WelchStats {
  double setup_ms = 0.0;
  double fft_ms = 0.0;
  double psd_ms = 0.0;
  double d2h_ms = 0.0;
  double total_s = 0.0;
};

// signal: 1-D float32 samples. noverlap < 0 means L / 2.
static WelchResult welch_psd(const std::vector<float>& signal, float fs,
                             int nperseg, int noverlap,
                             const std::string& window, bool onesided,
                             WelchStats* stats = nullptr) {
  const int L = nperseg;
  if (L < 2 || L > MAX_NPERSEG) {
    throw std::invalid_argument("nperseg must satisfy 2 <= nperseg <= 1024");
  }
  if ((L & (L - 1)) != 0) {
    throw std::invalid_argument("nperseg must be a power of two");
  }
  if (noverlap < 0) {
    noverlap = L / 2;
  }
  if (noverlap < 0 || noverlap >= L) {
    throw std::invalid_argument("noverlap must satisfy 0 <= noverlap < nperseg");
  }

  const int N = static_cast<int>(signal.size());
  if (N < L) {
    throw std::invalid_argument("signal is shorter than nperseg");
  }

  const int step = L - noverlap;
  const int n_seg = 1 + (N - L) / step;
  const int n_freq = onesided ? (L / 2 + 1) : L;
  const int nyquist_idx = (onesided && (L % 2 == 0)) ? (L / 2) : -1;

  const std::vector<float> w = make_window(window, L);
  double sum_w2 = 0.0;
  for (float value : w) {
    sum_w2 += static_cast<double>(value) * value;
  }
  const std::vector<int> bitrev = bit_reverse_table(L);
  std::vector<float> tw_cos, tw_sin;
  make_twiddles(L, tw_cos, tw_sin);

  const size_t signal_bytes = static_cast<size_t>(N) * sizeof(float);
  const size_t window_bytes = static_cast<size_t>(L) * sizeof(float);
  const size_t bitrev_bytes = static_cast<size_t>(L) * sizeof(int);
  const size_t twiddle_bytes = static_cast<size_t>(L / 2) * sizeof(float);
  const size_t spec_bytes = static_cast<size_t>(n_seg) * L * sizeof(float);
  const size_t psd_bytes = static_cast<size_t>(n_freq) * sizeof(float);

  float* d_signal = nullptr;
  float* d_window = nullptr;
  int* d_bitrev = nullptr;
  float* d_tw_cos = nullptr;
  float* d_tw_sin = nullptr;
  float* d_re = nullptr;
  float* d_im = nullptr;
  float* d_psd = nullptr;

  WelchStats local_stats;
  const auto setup_t0 = Clock::now();

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_signal), signal_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_window), window_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bitrev), bitrev_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tw_cos), twiddle_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tw_sin), twiddle_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_re), spec_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_im), spec_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_psd), psd_bytes));

  CUDA_CHECK(cudaMemcpy(d_signal, signal.data(), signal_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_window, w.data(), window_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bitrev, bitrev.data(), bitrev_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tw_cos, tw_cos.data(), twiddle_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tw_sin, tw_sin.data(), twiddle_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  local_stats.setup_ms = elapsed_ms(setup_t0);

  cudaEvent_t f0, f1, p0, p1;
  CUDA_CHECK(cudaEventCreate(&f0));
  CUDA_CHECK(cudaEventCreate(&f1));
  CUDA_CHECK(cudaEventCreate(&p0));
  CUDA_CHECK(cudaEventCreate(&p1));

  const auto total_t0 = Clock::now();

  // 1. Window + FFT, one block per segment (L threads, 2L floats shared)
  const size_t shared_bytes = 2 * static_cast<size_t>(L) * sizeof(float);
  CUDA_CHECK(cudaEventRecord(f0));
  welch_fft_kernel<<<n_seg, L, shared_bytes>>>(
      d_signal, d_window, d_bitrev, d_tw_cos, d_tw_sin, d_re, d_im, n_seg, L,
      step);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(f1));

  // 2. PSD averaging and scaling
  const float scale = static_cast<float>(1.0 / (static_cast<double>(n_seg) * fs * sum_w2));
  const int psd_blocks = (n_freq + THREADS_PSD - 1) / THREADS_PSD;
  CUDA_CHECK(cudaEventRecord(p0));
  welch_psd_kernel<<<psd_blocks, THREADS_PSD>>>(
      d_re, d_im, d_psd, n_seg, n_freq, L, scale, onesided ? 1 : 0,
      nyquist_idx);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(p1));

  CUDA_CHECK(cudaDeviceSynchronize());
  float phase_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&phase_ms, f0, f1));
  local_stats.fft_ms = phase_ms;
  CUDA_CHECK(cudaEventElapsedTime(&phase_ms, p0, p1));
  local_stats.psd_ms = phase_ms;

  // 3. Device-to-host
  WelchResult result;
  result.n_seg = n_seg;
  result.n_freq = n_freq;
  result.psd.resize(n_freq);
  const auto d2h_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(result.psd.data(), d_psd, psd_bytes,
                        cudaMemcpyDeviceToHost));
  local_stats.d2h_ms = elapsed_ms(d2h_t0);

  result.freqs.resize(n_freq);
  if (onesided) {
    for (int k = 0; k < n_freq; ++k) {
      result.freqs[k] = static_cast<float>(k) * fs / static_cast<float>(L);
    }
  } else {
    for (int k = 0; k < n_freq; ++k) {
      const int bin = (k <= L / 2) ? k : (k - L);
      result.freqs[k] = static_cast<float>(bin) * fs / static_cast<float>(L);
    }
  }

  local_stats.total_s = elapsed_s(total_t0);

  CUDA_CHECK(cudaEventDestroy(f0));
  CUDA_CHECK(cudaEventDestroy(f1));
  CUDA_CHECK(cudaEventDestroy(p0));
  CUDA_CHECK(cudaEventDestroy(p1));
  CUDA_CHECK(cudaFree(d_signal));
  CUDA_CHECK(cudaFree(d_window));
  CUDA_CHECK(cudaFree(d_bitrev));
  CUDA_CHECK(cudaFree(d_tw_cos));
  CUDA_CHECK(cudaFree(d_tw_sin));
  CUDA_CHECK(cudaFree(d_re));
  CUDA_CHECK(cudaFree(d_im));
  CUDA_CHECK(cudaFree(d_psd));

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

  constexpr float FS = 1000.0f;
  constexpr int N = 1 << 16;
  constexpr int L = 256;
  constexpr int NOVERLAP = L / 2;

  std::vector<float> signal(N);
  std::mt19937 gen(0);
  std::normal_distribution<float> noise(0.0f, 0.1f);
  for (int n = 0; n < N; ++n) {
    const float t = static_cast<float>(n) / FS;
    signal[n] = 0.7f * std::sin(2.0f * static_cast<float>(PI) * 50.0f * t) +
                0.5f * std::sin(2.0f * static_cast<float>(PI) * 120.0f * t) +
                noise(gen);
  }

  WelchStats stats;
  const WelchResult result =
      welch_psd(signal, FS, L, NOVERLAP, "hann", true, &stats);

  int peak = 0;
  for (int k = 1; k < result.n_freq; ++k) {
    if (result.psd[k] > result.psd[peak]) {
      peak = k;
    }
  }

  std::printf("N=%d, nperseg=%d, noverlap=%d, segments=%d\n", N, L, NOVERLAP,
              result.n_seg);
  std::printf("GPU total (kernels + D2H, excl. setup): %.4f s\n",
              stats.total_s);
  std::printf("  fft: %.3f ms | psd: %.3f ms | d2h: %.3f ms | setup: %.3f ms\n",
              stats.fft_ms, stats.psd_ms, stats.d2h_ms, stats.setup_ms);
  std::printf("peak frequency: %.2f Hz (psd=%.6g)\n", result.freqs[peak],
              result.psd[peak]);

  return EXIT_SUCCESS;
}
