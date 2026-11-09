#include <cuda_runtime.h>

#include <cstdint>

// GEGLU: split x into halves x1 = x[0..N/2), x2 = x[N/2..N) and compute
// out[i] = x1[i] * GELU(x2[i]), with GELU(y) = 0.5*y*(1 + erf(y/sqrt(2))).
//
// At the benchmark (N = 1,000,000, halfN = 500,000) the kernel reads 4 MB and
// writes 2 MB — 6 MB — so it is memory-bound: a same-traffic kernel measures
// 0.079 ms (76 GB/s) and the starter's scalar version already measures
// 0.079 ms.  The erff is hidden behind the loads.  What this version adds is
// the general-case shape: one float4 per thread when halfN % 4 == 0 and the
// pointers are 16-byte aligned, a scalar kernel otherwise.
//
// (This is the one activation challenge here where the math is free; contrast
// SiLU/SWiGLU at 400-600 KB, where the launch dominates and the division was
// worth removing.)

#ifndef THREADS
#define THREADS 256
#endif

__device__ __forceinline__ float gelu(float y) {
  constexpr float kInvSqrt2 = 0.7071067811865475f;
  return 0.5f * y * (1.0f + erff(y * kInvSqrt2));
}

__global__ void geglu_vec_kernel(const float4* __restrict__ x1,
                                 const float4* __restrict__ x2,
                                 float4* __restrict__ out, int n4) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    const float4 a = __ldg(&x1[i]);
    const float4 b = __ldg(&x2[i]);
    out[i] = make_float4(a.x * gelu(b.x), a.y * gelu(b.y), a.z * gelu(b.z),
                         a.w * gelu(b.w));
  }
}

// General path: any halfN, any pointer alignment.
__global__ void geglu_scalar_kernel(const float* __restrict__ x1,
                                    const float* __restrict__ x2,
                                    float* __restrict__ out, int halfN) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < halfN) out[i] = __ldg(&x1[i]) * gelu(__ldg(&x2[i]));
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  if (N <= 0) return;
  const int halfN = N / 2;
  if (halfN <= 0) return;
  const float* x2 = input + halfN;

  const bool aligned =
      (((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) &
        15u) == 0) &&
      ((halfN & 3) == 0);
  if (aligned) {
    const int n4 = halfN >> 2;
    const int blocks = (n4 + THREADS - 1) / THREADS;
    geglu_vec_kernel<<<blocks, THREADS>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<const float4*>(x2), reinterpret_cast<float4*>(output),
        n4);
  } else {
    const int blocks = (halfN + THREADS - 1) / THREADS;
    geglu_scalar_kernel<<<blocks, THREADS>>>(input, x2, output, halfN);
  }
  cudaDeviceSynchronize();
}
