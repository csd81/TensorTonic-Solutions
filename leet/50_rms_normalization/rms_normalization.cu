#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cmath>
#include <cstdint>

namespace cg = cooperative_groups;

// RMS normalization: y_i = gamma * x_i / sqrt(mean(x^2) + eps) + beta.
//
// At the benchmark size (N = 100,000, 400 KB of input) the problem is launch-
// and latency-bound, not bandwidth-bound, so the implementation minimises the
// number of launches and allocations:
//
//   * a single cooperative kernel does the sum of squares, a grid-wide
//     barrier (cg::this_grid().sync()) and then the scale-and-shift pass --
//     one launch instead of two,
//   * a cached 4-byte device scratch replaces per-call cudaMalloc / cudaFree
//     (which forces a device synchronisation and costs more than the kernels
//     at this size),
//   * the reduction uses float4 loads (one thread per four elements), so only
//     a few dozen atomics reach the counter,
//   * the grid is capped to the number of co-resident blocks reported by the
//     occupancy API, as a cooperative launch requires; if the device does not
//     support cooperative launch the two-kernel path is used instead.

#define THREADS 256

__device__ __forceinline__ float block_reduce_sum(float acc, float* s_warp,
                                                  int tid, int lane, int warp) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) s_warp[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < THREADS / 32) ? s_warp[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;  // valid in lane 0
  }
  return 0.0f;
}

__device__ __forceinline__ float rms_scale(const float* sum_sq, int N,
                                           float gamma, float eps) {
  const float mean_sq = sum_sq[0] / static_cast<float>(N);
  return rsqrtf(mean_sq + eps) * gamma;
}

// --- single cooperative kernel -------------------------------------------
__global__ void rms_coop_kernel(const float* __restrict__ x,
                                float* __restrict__ y,
                                float* __restrict__ sum_sq, float gamma,
                                float beta, int N, float eps) {
  __shared__ float s_warp[THREADS / 32];
  __shared__ float s_scale;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int stride = gridDim.x * blockDim.x;
  const int n4 = N >> 2;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  float4* y4 = reinterpret_cast<float4*>(y);

  float acc = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += stride) {
    const float4 v = x4[i];
    acc = fmaf(v.x, v.x, acc);
    acc = fmaf(v.y, v.y, acc);
    acc = fmaf(v.z, v.z, acc);
    acc = fmaf(v.w, v.w, acc);
  }
  if (blockIdx.x == 0 && tid < (N & 3)) {
    const float v = x[(n4 << 2) + tid];
    acc = fmaf(v, v, acc);
  }
  const float block_sum = block_reduce_sum(acc, s_warp, tid, lane, warp);
  if (warp == 0 && lane == 0) atomicAdd(sum_sq, block_sum);

  cg::this_grid().sync();

  if (tid == 0) s_scale = rms_scale(sum_sq, N, gamma, eps);
  __syncthreads();
  const float scale = s_scale;

  for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += stride) {
    const float4 v = x4[i];
    y4[i] = make_float4(fmaf(v.x, scale, beta), fmaf(v.y, scale, beta),
                        fmaf(v.z, scale, beta), fmaf(v.w, scale, beta));
  }
  if (blockIdx.x == 0 && tid < (N & 3)) {
    const int idx = (n4 << 2) + tid;
    y[idx] = fmaf(x[idx], scale, beta);
  }
}

// --- two-kernel fallback --------------------------------------------------
__global__ void sumsq_vec4_kernel(const float* __restrict__ x,
                                  const float4* __restrict__ x4,
                                  float* __restrict__ sum_sq, int N) {
  __shared__ float s_warp[THREADS / 32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int stride = gridDim.x * blockDim.x;
  const int n4 = N >> 2;

  float acc = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += stride) {
    const float4 v = x4[i];
    acc = fmaf(v.x, v.x, acc);
    acc = fmaf(v.y, v.y, acc);
    acc = fmaf(v.z, v.z, acc);
    acc = fmaf(v.w, v.w, acc);
  }
  if (blockIdx.x == 0 && tid < (N & 3)) {
    const float v = x[(n4 << 2) + tid];
    acc = fmaf(v, v, acc);
  }
  const float block_sum = block_reduce_sum(acc, s_warp, tid, lane, warp);
  if (warp == 0 && lane == 0) atomicAdd(sum_sq, block_sum);
}

__global__ void norm_vec4_kernel(const float* __restrict__ x,
                                 const float4* __restrict__ x4,
                                 float* __restrict__ output,
                                 float4* __restrict__ out4,
                                 const float* __restrict__ sum_sq, float gamma,
                                 float beta, int N, float eps) {
  __shared__ float s_scale;
  if (threadIdx.x == 0) s_scale = rms_scale(sum_sq, N, gamma, eps);
  __syncthreads();
  const float scale = s_scale;

  const int n4 = N >> 2;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 v = x4[i];
    out4[i] = make_float4(fmaf(v.x, scale, beta), fmaf(v.y, scale, beta),
                          fmaf(v.z, scale, beta), fmaf(v.w, scale, beta));
  }
  if (blockIdx.x == 0 && threadIdx.x < (N & 3)) {
    const int idx = (n4 << 2) + threadIdx.x;
    output[idx] = fmaf(x[idx], scale, beta);
  }
}

__global__ void sumsq_scalar_kernel(const float* __restrict__ x,
                                    float* __restrict__ sum_sq, int N) {
  __shared__ float s_warp[THREADS / 32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int stride = gridDim.x * blockDim.x;

  float acc = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < N; i += stride) {
    const float v = x[i];
    acc = fmaf(v, v, acc);
  }
  const float block_sum = block_reduce_sum(acc, s_warp, tid, lane, warp);
  if (warp == 0 && lane == 0) atomicAdd(sum_sq, block_sum);
}

__global__ void norm_scalar_kernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   const float* __restrict__ sum_sq,
                                   float gamma, float beta, int N, float eps) {
  __shared__ float s_scale;
  if (threadIdx.x == 0) s_scale = rms_scale(sum_sq, N, gamma, eps);
  __syncthreads();
  const float scale = s_scale;

  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
    y[i] = fmaf(x[i], scale, beta);
  }
}

// input, output are device pointers.
extern "C" void solve(const float* input, float gamma, float beta, float* output,
                      int N, float eps) {
  if (input == nullptr || output == nullptr || N <= 0) return;

  // Cached 4-byte scratch; allocating it per call costs more than the kernels.
  static float* d_sum_sq = nullptr;
  if (d_sum_sq == nullptr) cudaMalloc(&d_sum_sq, sizeof(float));
  cudaMemsetAsync(d_sum_sq, 0, sizeof(float));

  const bool aligned = ((reinterpret_cast<uintptr_t>(input) |
                         reinterpret_cast<uintptr_t>(output)) &
                        15u) == 0;

  // Co-resident block capacity for the cooperative launch (computed once).
  static int coop_capacity = -2;
  if (coop_capacity == -2) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    coop_capacity = -1;
    if (prop.cooperativeLaunch) {
      int per_sm = 0;
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &per_sm, rms_coop_kernel, THREADS, 0) == cudaSuccess &&
          per_sm > 0) {
        coop_capacity = per_sm * prop.multiProcessorCount;
      }
    }
  }

  if (aligned && coop_capacity > 0) {
    int blocks = (N / 4 + THREADS - 1) / THREADS;
    if (blocks < 1) blocks = 1;
    if (blocks > coop_capacity) blocks = coop_capacity;
    void* args[] = {(void*)&input, (void*)&output, (void*)&d_sum_sq,
                    (void*)&gamma, (void*)&beta, (void*)&N, (void*)&eps};
    const cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)rms_coop_kernel, dim3(blocks), dim3(THREADS), args, 0, 0);
    if (err == cudaSuccess) {
      cudaDeviceSynchronize();
      return;
    }
    // Fall through to the two-kernel path if the cooperative launch is
    // rejected; the output has not been touched.
  }

  if (aligned) {
    const int n4 = N >> 2;
    int blocks = (n4 + THREADS - 1) / THREADS;
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;
    sumsq_vec4_kernel<<<blocks, THREADS>>>(
        input, reinterpret_cast<const float4*>(input), d_sum_sq, N);
    norm_vec4_kernel<<<blocks, THREADS>>>(
        input, reinterpret_cast<const float4*>(input), output,
        reinterpret_cast<float4*>(output), d_sum_sq, gamma, beta, N, eps);
  } else {
    int blocks = (N + THREADS - 1) / THREADS;
    if (blocks > 1024) blocks = 1024;
    sumsq_scalar_kernel<<<blocks, THREADS>>>(input, d_sum_sq, N);
    norm_scalar_kernel<<<blocks, THREADS>>>(input, output, d_sum_sq, gamma,
                                            beta, N, eps);
  }
  cudaDeviceSynchronize();
}
