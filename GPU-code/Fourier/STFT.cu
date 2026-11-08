// STFT.cu -- CUDA C++ port of Fourier/STFT.py.
//
// Batched short-time Fourier transform, one CUDA block per frame:
//   * load and window one frame into shared memory (zero-padded to n_fft),
//   * compute the non-negative-frequency DFT bins with a trigonometric
//     recurrence (no sin/cos in the inner loop),
//   * scale by 1 / sum(window), matching scipy.signal.stft(scaling="spectrum").
//
// Output layout is [n_freq_bins][n_frames], the same as SciPy.
// The reference in this program is a sequential double-precision direct DFT
// (not SciPy), so the speedup compares the same algorithm on CPU and GPU.
//
// Build:
//   nvcc -O3 -arch=native -o STFT STFT.cu
// Run:
//   ./STFT
//
// Constraints: frame_len <= n_fft <= 1024, 1 <= hop_size <= frame_len.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#define MAX_FFT 1024

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

// ---------------------------------------------------------------------------
// GPU kernel
// ---------------------------------------------------------------------------

__global__ void stft_kernel(const float* __restrict__ signal,
                            const float* __restrict__ window,
                            float* __restrict__ out_real,
                            float* __restrict__ out_imag, int signal_len,
                            int frame_len, int hop_size, int n_frames,
                            int n_fft, float scale) {
  __shared__ float s_frame[MAX_FFT];

  const int frame_idx = blockIdx.x;
  const int tx = threadIdx.x;
  if (frame_idx >= n_frames) {
    return;
  }

  const int frame_start = frame_idx * hop_size;

  // Load and window, zero-padding up to n_fft.
  for (int idx = tx; idx < n_fft; idx += blockDim.x) {
    if (idx < frame_len && (frame_start + idx) < signal_len) {
      s_frame[idx] = signal[frame_start + idx] * window[idx];
    } else {
      s_frame[idx] = 0.0f;
    }
  }
  __syncthreads();

  int out_idx = frame_idx;

  for (int m = tx; m < n_fft / 2 + 1; m += blockDim.x) {
    const double angle_step =
        -2.0 * 3.14159265358979323846 * m / static_cast<double>(n_fft);
    double cos_step = std::cos(angle_step);
    double sin_step = std::sin(angle_step);

    double c = 1.0;
    double s = 0.0;
    double sum_real = 0.0;
    double sum_imag = 0.0;

    for (int n = 0; n < n_fft; ++n) {
      const double val = static_cast<double>(s_frame[n]);
      sum_real += val * c;
      sum_imag += val * s;

      const double c_new = c * cos_step - s * sin_step;
      const double s_new = s * cos_step + c * sin_step;
      c = c_new;
      s = s_new;
    }

    out_real[static_cast<size_t>(m) * n_frames + out_idx] =
        static_cast<float>(sum_real * static_cast<double>(scale));
    out_imag[static_cast<size_t>(m) * n_frames + out_idx] =
        static_cast<float>(sum_imag * static_cast<double>(scale));
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct StftOutput {
  int n_freq = 0;
  int n_frames = 0;
  std::vector<float> real;
  std::vector<float> imag;
};

struct StftStats {
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
};

static void validate(const std::vector<float>& signal,
                     const std::vector<float>& window, int hop_size,
                     int n_fft) {
  if (signal.empty()) {
    throw std::invalid_argument("signal must not be empty");
  }
  if (window.empty()) {
    throw std::invalid_argument("window must not be empty");
  }
  const int frame_len = static_cast<int>(window.size());
  if (frame_len > n_fft || n_fft > MAX_FFT) {
    throw std::invalid_argument(
        "require frame_len <= n_fft <= 1024");
  }
  if (hop_size < 1 || hop_size > frame_len) {
    throw std::invalid_argument(
        "require 1 <= hop_size <= frame_len");
  }
  if (static_cast<int>(signal.size()) < frame_len) {
    throw std::invalid_argument("signal is shorter than the window");
  }
}

static StftOutput run_stft_gpu(const std::vector<float>& signal,
                               const std::vector<float>& window, int hop_size,
                               int n_fft, StftStats* stats) {
  validate(signal, window, hop_size, n_fft);

  const int signal_len = static_cast<int>(signal.size());
  const int frame_len = static_cast<int>(window.size());
  const int n_frames = (signal_len - frame_len) / hop_size + 1;
  const int n_freq = n_fft / 2 + 1;

  double window_sum = 0.0;
  for (float value : window) {
    window_sum += static_cast<double>(value);
  }
  if (window_sum == 0.0) {
    throw std::invalid_argument("window sum must be non-zero");
  }
  const float scale = static_cast<float>(1.0 / window_sum);

  const size_t signal_bytes = static_cast<size_t>(signal_len) * sizeof(float);
  const size_t window_bytes = static_cast<size_t>(frame_len) * sizeof(float);
  const size_t out_count = static_cast<size_t>(n_freq) * n_frames;
  const size_t out_bytes = out_count * sizeof(float);

  float* d_signal = nullptr;
  float* d_window = nullptr;
  float* d_real = nullptr;
  float* d_imag = nullptr;

  StftStats local;
  const auto h2d_t0 = Clock::now();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_signal), signal_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_window), window_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_real), out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_imag), out_bytes));
  CUDA_CHECK(cudaMemcpy(d_signal, signal.data(), signal_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_window, window.data(), window_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  local.h2d_ms = elapsed_ms(h2d_t0);

  const int threads = std::min(256, n_fft);
  const int blocks = n_frames;

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start));

  stft_kernel<<<blocks, threads>>>(d_signal, d_window, d_real, d_imag,
                                   signal_len, frame_len, hop_size, n_frames,
                                   n_fft, scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, end));
  local.kernel_ms = kernel_ms;

  StftOutput out;
  out.n_freq = n_freq;
  out.n_frames = n_frames;
  out.real.resize(out_count);
  out.imag.resize(out_count);

  const auto d2h_t0 = Clock::now();
  CUDA_CHECK(cudaMemcpy(out.real.data(), d_real, out_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.imag.data(), d_imag, out_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  local.d2h_ms = elapsed_ms(d2h_t0);
  local.total_ms = local.h2d_ms + local.kernel_ms + local.d2h_ms;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(d_signal));
  CUDA_CHECK(cudaFree(d_window));
  CUDA_CHECK(cudaFree(d_real));
  CUDA_CHECK(cudaFree(d_imag));

  if (stats != nullptr) {
    *stats = local;
  }
  return out;
}

// Sequential double-precision reference of the same direct DFT.
static StftOutput run_stft_cpu(const std::vector<float>& signal,
                               const std::vector<float>& window, int hop_size,
                               int n_fft) {
  validate(signal, window, hop_size, n_fft);

  const int signal_len = static_cast<int>(signal.size());
  const int frame_len = static_cast<int>(window.size());
  const int n_frames = (signal_len - frame_len) / hop_size + 1;
  const int n_freq = n_fft / 2 + 1;

  double window_sum = 0.0;
  for (float value : window) {
    window_sum += static_cast<double>(value);
  }
  if (window_sum == 0.0) {
    throw std::invalid_argument("window sum must be non-zero");
  }
  const double scale = 1.0 / window_sum;

  StftOutput out;
  out.n_freq = n_freq;
  out.n_frames = n_frames;
  out.real.assign(static_cast<size_t>(n_freq) * n_frames, 0.0f);
  out.imag.assign(static_cast<size_t>(n_freq) * n_frames, 0.0f);

  std::vector<double> frame(n_fft, 0.0);
  for (int f = 0; f < n_frames; ++f) {
    const int start = f * hop_size;
    for (int i = 0; i < n_fft; ++i) {
      if (i < frame_len && (start + i) < signal_len) {
        frame[i] = static_cast<double>(signal[start + i]) *
                   static_cast<double>(window[i]);
      } else {
        frame[i] = 0.0;
      }
    }

    for (int m = 0; m < n_freq; ++m) {
      const double angle_step =
          -2.0 * 3.14159265358979323846 * m / static_cast<double>(n_fft);
      const double cos_step = std::cos(angle_step);
      const double sin_step = std::sin(angle_step);
      double c = 1.0;
      double s = 0.0;
      double sum_real = 0.0;
      double sum_imag = 0.0;
      for (int n = 0; n < n_fft; ++n) {
        sum_real += frame[n] * c;
        sum_imag += frame[n] * s;
        const double c_new = c * cos_step - s * sin_step;
        const double s_new = s * cos_step + c * sin_step;
        c = c_new;
        s = s_new;
      }
      out.real[static_cast<size_t>(m) * n_frames + f] =
          static_cast<float>(sum_real * scale);
      out.imag[static_cast<size_t>(m) * n_frames + f] =
          static_cast<float>(sum_imag * scale);
    }
  }
  return out;
}

static void compare(const StftOutput& a, const StftOutput& b, float* max_abs,
                    double* rel_frob) {
  if (a.n_freq != b.n_freq || a.n_frames != b.n_frames) {
    throw std::runtime_error("STFT shape mismatch");
  }
  double max_diff = 0.0;
  double diff2 = 0.0;
  double norm2 = 0.0;
  for (size_t i = 0; i < a.real.size(); ++i) {
    const double dr = static_cast<double>(a.real[i]) - b.real[i];
    const double di = static_cast<double>(a.imag[i]) - b.imag[i];
    max_diff = std::max(max_diff, std::max(std::fabs(dr), std::fabs(di)));
    diff2 += dr * dr + di * di;
    norm2 += static_cast<double>(b.real[i]) * b.real[i] +
             static_cast<double>(b.imag[i]) * b.imag[i];
  }
  *max_abs = static_cast<float>(max_diff);
  *rel_frob = std::sqrt(diff2 / norm2);
}

static std::vector<float> make_hann(int frame_len) {
  std::vector<float> win(frame_len);
  for (int n = 0; n < frame_len; ++n) {
    win[n] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * n / frame_len));
  }
  return win;
}

template <typename T>
static T median(std::vector<T> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\n", prop.name);

  const int sr = 44100;
  const int frame_len = 512;
  const int hop = 256;
  const int n_fft = 512;
  const std::vector<float> window = make_hann(frame_len);

  // ---------------------------------------------------------------------
  // Numerical verification on a 2-second multi-tone signal
  // ---------------------------------------------------------------------
  const int n_verify = 2 * sr;
  std::vector<float> test_signal(n_verify);
  for (int i = 0; i < n_verify; ++i) {
    const double t = static_cast<double>(i) / sr;
    test_signal[i] = static_cast<float>(
        std::sin(2.0 * 3.14159265358979323846 * 440.0 * t) +
        0.5 * std::sin(2.0 * 3.14159265358979323846 * 1200.0 * t));
  }

  // Warm-up (CUDA context/allocator)
  const std::vector<float> warm_signal(test_signal.begin(),
                                       test_signal.begin() + 4 * frame_len);
  run_stft_gpu(warm_signal, window, hop, n_fft, nullptr);

  const StftOutput gpu_verify = run_stft_gpu(test_signal, window, hop, n_fft,
                                             nullptr);
  const StftOutput cpu_verify = run_stft_cpu(test_signal, window, hop, n_fft);
  float verify_max = 0.0f;
  double verify_rel = 0.0;
  compare(gpu_verify, cpu_verify, &verify_max, &verify_rel);

  std::printf("\n=== NUMERIKUS VERIFIKACIO ===\n");
  std::printf("Kimeneti matrix: (%d, %d) (frekvenciabin x idokeret)\n",
              gpu_verify.n_freq, gpu_verify.n_frames);
  std::printf("Maximalis abszolut hiba: %.6e\n", verify_max);
  std::printf("Relativ Frobenius-hiba:  %.6e\n", verify_rel);
  std::printf("Tolerancian belul: %s\n\n",
              verify_max < 1e-4f ? "IGEN" : "NEM");

  // ---------------------------------------------------------------------
  // Benchmark
  // ---------------------------------------------------------------------
  std::printf("=== TELJESITMENYMERES (BENCHMARK) ===\n");
  std::printf("%-10s %-10s %-14s %-16s %-16s %-12s\n", "Hossz(mp)", "Mintak",
              "CPU DFT (ms)", "GPU kernel (ms)", "GPU total (ms)", "Speedup");

  for (int sec : {2, 5, 10}) {
    const int n = sec * sr;
    std::mt19937 gen(42 + sec);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> signal(n);
    for (float& value : signal) {
      value = normal(gen);
    }

    const auto cpu_t0 = Clock::now();
    const StftOutput cpu = run_stft_cpu(signal, window, hop, n_fft);
    const double cpu_ms = elapsed_ms(cpu_t0);

    StftOutput gpu;
    std::vector<double> kernel_times;
    std::vector<double> total_times;
    float max_err = 0.0f;
    double rel_err = 0.0;
    for (int r = 0; r < 5; ++r) {
      StftStats stats;
      gpu = run_stft_gpu(signal, window, hop, n_fft, &stats);
      kernel_times.push_back(stats.kernel_ms);
      total_times.push_back(stats.total_ms);
    }
    compare(gpu, cpu, &max_err, &rel_err);

    const double kernel_ms = median(kernel_times);
    const double total_ms = median(total_times);
    const double speedup = cpu_ms / kernel_ms;
    std::printf("%-10d %-10d %-14.2f %-16.2f %-16.2f %-12.2fx\n", sec, n,
                cpu_ms, kernel_ms, total_ms, speedup);
    std::printf("  hiba: max=%.2e, rel=%.2e\n", max_err, rel_err);
  }

  return EXIT_SUCCESS;
}
