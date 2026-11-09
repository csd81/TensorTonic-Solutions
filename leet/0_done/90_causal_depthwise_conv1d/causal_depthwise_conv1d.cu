#include <cuda_runtime.h>

#include <cstdint>

// Causal depthwise 1D convolution over a channels-last (B, L, D) tensor:
//
//   out[b, l, d] = bias[d] + sum_{k<K} weight[d, k] * x[b, l-k, d]
//
// with x[b, l-k, d] = 0 for l-k < 0.
//
// The starter already used the right shape: one thread owns four channels
// (float4) and a tile of TILE_L sequence positions, keeping a sliding window
// in registers so each input is loaded once.  This version keeps that design
// but:
//   * templates on K (1..8) instead of masking with a compile-time MAX_K=8,
//     halving the live window/weight registers at K=4;
//   * replaces the window shift (K-1 register moves per step, ~100 moves per
//     tile) with a circular buffer indexed by (s - j) mod K, where both
//     indices are compile-time constants in the unrolled step loop -- no
//     moves at all;
//   * raises the tile from 8 to TILE_L sequence positions, cutting the halo
//     re-read from (TILE+K-1)/TILE = 1.38x to 1.09x at K=4.

#ifndef TILE_L
#define TILE_L 256
#endif
#ifndef THREADS
#define THREADS 256
#endif

// ---------------------------------------------------------------- vec4 path

template <int K, int TILE>
__global__ void conv_vec4_kernel(const float4* __restrict__ x,
                                 const float* __restrict__ weight,
                                 const float4* __restrict__ bias,
                                 float4* __restrict__ out, int L, int D4) {
  const int d4 = blockIdx.x * blockDim.x + threadIdx.x;
  if (d4 >= D4) return;
  const int b = blockIdx.z;
  const int l0 = blockIdx.y * TILE;
  const int d = d4 * 4;

  // weight[d..d+3][0..K-1], one float4 per tap (4 channels each).
  float4 wv[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    wv[k] = make_float4(weight[(d + 0) * K + k], weight[(d + 1) * K + k],
                        weight[(d + 2) * K + k], weight[(d + 3) * K + k]);
  }
  const float4 bv = bias[d4];

  const float4* xb = x + (static_cast<size_t>(b) * L + l0) * D4 + d4;
  float4* ob = out + (static_cast<size_t>(b) * L + l0) * D4 + d4;

  // Circular window: slot (s mod K) holds x[l0 + s]; the (K-1) slots before
  // the tile are pre-filled with x[l0 - j] (zero when l0 - j < 0).
  float4 win[K];
#pragma unroll
  for (int j = 0; j < K; ++j) win[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int j = 1; j < K; ++j) {
    if (l0 - j >= 0) win[(K - j) % K] = xb[-static_cast<size_t>(j) * D4];
  }

#pragma unroll
  for (int s = 0; s < TILE; ++s) {
    const int l = l0 + s;
    if (l < L) {
      win[s % K] = xb[static_cast<size_t>(s) * D4];
      float4 acc = bv;
#pragma unroll
      for (int j = 0; j < K; ++j) {
        const float4 t = win[((s - j) % K + K) % K];
        acc.x = fmaf(wv[j].x, t.x, acc.x);
        acc.y = fmaf(wv[j].y, t.y, acc.y);
        acc.z = fmaf(wv[j].z, t.z, acc.z);
        acc.w = fmaf(wv[j].w, t.w, acc.w);
      }
      ob[static_cast<size_t>(s) * D4] = acc;
    }
  }
}

// -------------------------------------------------------------- scalar path

template <int K, int TILE>
__global__ void conv_scalar_kernel(const float* __restrict__ x,
                                   const float* __restrict__ weight,
                                   const float* __restrict__ bias,
                                   float* __restrict__ out, int L, int D) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= D) return;
  const int b = blockIdx.z;
  const int l0 = blockIdx.y * TILE;

  float wv[K];
#pragma unroll
  for (int k = 0; k < K; ++k) wv[k] = weight[d * K + k];
  const float bv = bias[d];

  const float* xb = x + (static_cast<size_t>(b) * L + l0) * D + d;
  float* ob = out + (static_cast<size_t>(b) * L + l0) * D + d;

  float win[K];
#pragma unroll
  for (int j = 0; j < K; ++j) win[j] = 0.0f;
#pragma unroll
  for (int j = 1; j < K; ++j) {
    if (l0 - j >= 0) win[(K - j) % K] = xb[-static_cast<size_t>(j) * D];
  }

#pragma unroll
  for (int s = 0; s < TILE; ++s) {
    const int l = l0 + s;
    if (l < L) {
      win[s % K] = xb[static_cast<size_t>(s) * D];
      float acc = bv;
#pragma unroll
      for (int j = 0; j < K; ++j) {
        acc = fmaf(wv[j], win[((s - j) % K + K) % K], acc);
      }
      ob[static_cast<size_t>(s) * D] = acc;
    }
  }
}

// -------------------------------------------------------------------- launch

template <int K>
static void launch(const float* x, const float* weight, const float* bias,
                   float* output, int B, int L, int D, bool vec4) {
  const int grid_y = (L + TILE_L - 1) / TILE_L;
  if (vec4) {
    const int D4 = D >> 2;
    dim3 grid((D4 + THREADS - 1) / THREADS, grid_y, B);
    conv_vec4_kernel<K, TILE_L><<<grid, THREADS>>>(
        reinterpret_cast<const float4*>(x), weight,
        reinterpret_cast<const float4*>(bias),
        reinterpret_cast<float4*>(output), L, D4);
  } else {
    dim3 grid((D + THREADS - 1) / THREADS, grid_y, B);
    conv_scalar_kernel<K, TILE_L><<<grid, THREADS>>>(x, weight, bias, output,
                                                     L, D);
  }
}

// x, weight, bias, output are device pointers.
extern "C" void solve(const float* x, const float* weight, const float* bias,
                      float* output, int B, int L, int D, int K) {
  if (B <= 0 || L <= 0 || D <= 0 || K <= 0 || K > 8) return;

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(bias) |
        reinterpret_cast<uintptr_t>(output)) &
       15u) == 0;
  const bool vec4 = (D % 4 == 0) && aligned;

  switch (K) {
    case 1: launch<1>(x, weight, bias, output, B, L, D, vec4); break;
    case 2: launch<2>(x, weight, bias, output, B, L, D, vec4); break;
    case 3: launch<3>(x, weight, bias, output, B, L, D, vec4); break;
    case 4: launch<4>(x, weight, bias, output, B, L, D, vec4); break;
    case 5: launch<5>(x, weight, bias, output, B, L, D, vec4); break;
    case 6: launch<6>(x, weight, bias, output, B, L, D, vec4); break;
    case 7: launch<7>(x, weight, bias, output, B, L, D, vec4); break;
    default: launch<8>(x, weight, bias, output, B, L, D, vec4); break;
  }
  cudaDeviceSynchronize();
}
