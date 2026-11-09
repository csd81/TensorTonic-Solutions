#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

// Multi-head self-attention:
//   head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i,  output = Concat(head_1..head_h)
// where Q_i, K_i, V_i are the d_k-wide column slices of Q, K, V belonging to
// head i (d_k = d_model / h), and each row is an independent softmax.
//
// Fast path (d_k <= 128) -- flash-attention style, one pass over N:
//   * one warp owns ROWS_PER_WARP (= 2) rows of the head's output,
//   * the 32 lanes of a warp are the 32 keys of the current K/V tile
//     (BLOCK_N == warp size), so each lane computes one full score per row
//     with a plain dot product -- no cross-lane reduction for the scores,
//   * the running max / sum are warp-wide shuffles,
//   * K/V tiles for the head are staged in shared memory (padded by one
//     float so sK[lane][k] and sV[j][lane] are bank-conflict free) and the
//     next tile is prefetched into registers while the current one is used.
//
// Slow path (d_k > 128) -- the original block-per-(head, row) kernel with
// block-wide dot-product reductions; kept as a correct fallback for heads
// wider than the shared-memory tile.

#define MAX_DK 128        // fast-path limit on d_k
#define BLOCK_N 32        // keys per tile == warp size
#define ROWS_PER_WARP 2   // rows of the head handled by one warp
#define WARPS_PER_BLOCK 8
#define BLOCK_M (ROWS_PER_WARP * WARPS_PER_BLOCK)  // 16 rows per block

__device__ __forceinline__ float warp_allreduce_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

__device__ __forceinline__ float warp_allreduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

__global__ void mha_flash_kernel(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 float* __restrict__ output, int N, int d_model,
                                 int d_k, float scale) {
  __shared__ float sQ[BLOCK_M][MAX_DK + 1];
  __shared__ float sK[BLOCK_N][MAX_DK + 1];
  __shared__ float sV[BLOCK_N][MAX_DK + 1];
  __shared__ float sP[BLOCK_M][BLOCK_N];  // softmax numerators per warp tile

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int head_off = blockIdx.y * d_k;  // first column of this head

  const int row0 = blockIdx.x * BLOCK_M + warp * ROWS_PER_WARP;
  const int row1 = row0 + 1;
  const bool valid0 = row0 < N;
  const bool valid1 = row1 < N;

  const int q0 = warp * ROWS_PER_WARP;  // shared row indices of this warp
  const int q1 = q0 + 1;

  // Stage this warp's Q rows (head slice) in shared memory.
  for (int k = lane; k < d_k; k += 32) {
    sQ[q0][k] = valid0 ? Q[row0 * d_model + head_off + k] : 0.0f;
    sQ[q1][k] = valid1 ? Q[row1 * d_model + head_off + k] : 0.0f;
  }

  // Per-row online-softmax state.
  float m0 = -FLT_MAX, l0 = 0.0f;
  float m1 = -FLT_MAX, l1 = 0.0f;
  // Each lane owns elements lane, lane+32, lane+64, lane+96 of the output row.
  float o00 = 0.0f, o01 = 0.0f, o02 = 0.0f, o03 = 0.0f;
  float o10 = 0.0f, o11 = 0.0f, o12 = 0.0f, o13 = 0.0f;

  // Software pipeline: tile t+1 is fetched into registers while tile t is
  // being computed, hiding the global-memory latency.
  float kbuf[4][4], vbuf[4][4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int j = warp * 4 + r;
    const int gj = j;  // first tile: k_start = 0
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = lane + 32 * i;
      const bool ok = (gj < N) && (k < d_k);
      kbuf[r][i] = ok ? K[gj * d_model + head_off + k] : 0.0f;
      vbuf[r][i] = ok ? V[gj * d_model + head_off + k] : 0.0f;
    }
  }

  for (int k_start = 0; k_start < N; k_start += BLOCK_N) {
    // Publish the staged tile, then start loading the next one.
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int j = warp * 4 + r;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k = lane + 32 * i;
        if (k < d_k) {
          sK[j][k] = kbuf[r][i];
          sV[j][k] = vbuf[r][i];
        }
      }
    }
    __syncthreads();

    const int next = k_start + BLOCK_N;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int j = warp * 4 + r;
      const int gj = next + j;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k = lane + 32 * i;
        const bool ok = (next < N) && (gj < N) && (k < d_k);
        kbuf[r][i] = ok ? K[gj * d_model + head_off + k] : 0.0f;
        vbuf[r][i] = ok ? V[gj * d_model + head_off + k] : 0.0f;
      }
    }

    // Scores for the keys owned by each lane (one key per lane per row).
    float dot0 = 0.0f, dot1 = 0.0f;
    for (int k = 0; k < d_k; ++k) {
      const float kv = sK[lane][k];
      dot0 += sQ[q0][k] * kv;
      dot1 += sQ[q1][k] * kv;
    }
    float s0 = dot0 * scale;
    float s1 = dot1 * scale;
    if (k_start + lane >= N) {  // padding keys never contribute
      s0 = -FLT_MAX;
      s1 = -FLT_MAX;
    }

    // Online softmax update (warp-wide, every lane keeps the result).
    const float mn0 = fmaxf(m0, warp_allreduce_max(s0));
    const float mn1 = fmaxf(m1, warp_allreduce_max(s1));
    const float a0 = expf(m0 - mn0);
    const float a1 = expf(m1 - mn1);
    const float p0 = expf(s0 - mn0);
    const float p1 = expf(s1 - mn1);
    l0 = l0 * a0 + warp_allreduce_sum(p0);
    l1 = l1 * a1 + warp_allreduce_sum(p1);
    m0 = mn0;
    m1 = mn1;

    o00 *= a0; o01 *= a0; o02 *= a0; o03 *= a0;
    o10 *= a1; o11 *= a1; o12 *= a1; o13 *= a1;

    // Publish this tile's numerators and accumulate P*V for both rows.
    sP[q0][lane] = p0;
    sP[q1][lane] = p1;
    __syncwarp();
#pragma unroll
    for (int j = 0; j < BLOCK_N; ++j) {
      const float pj0 = sP[q0][j];
      const float pj1 = sP[q1][j];
      int idx = lane;
      if (idx < d_k) {
        const float v = sV[j][idx];
        o00 += pj0 * v;
        o10 += pj1 * v;
      }
      idx += 32;
      if (idx < d_k) {
        const float v = sV[j][idx];
        o01 += pj0 * v;
        o11 += pj1 * v;
      }
      idx += 32;
      if (idx < d_k) {
        const float v = sV[j][idx];
        o02 += pj0 * v;
        o12 += pj1 * v;
      }
      idx += 32;
      if (idx < d_k) {
        const float v = sV[j][idx];
        o03 += pj0 * v;
        o13 += pj1 * v;
      }
    }
    __syncthreads();
  }

  const float inv0 = 1.0f / l0;
  const float inv1 = 1.0f / l1;
  int idx = lane;
  if (idx < d_k) {
    if (valid0) output[row0 * d_model + head_off + idx] = o00 * inv0;
    if (valid1) output[row1 * d_model + head_off + idx] = o10 * inv1;
  }
  idx += 32;
  if (idx < d_k) {
    if (valid0) output[row0 * d_model + head_off + idx] = o01 * inv0;
    if (valid1) output[row1 * d_model + head_off + idx] = o11 * inv1;
  }
  idx += 32;
  if (idx < d_k) {
    if (valid0) output[row0 * d_model + head_off + idx] = o02 * inv0;
    if (valid1) output[row1 * d_model + head_off + idx] = o12 * inv1;
  }
  idx += 32;
  if (idx < d_k) {
    if (valid0) output[row0 * d_model + head_off + idx] = o03 * inv0;
    if (valid1) output[row1 * d_model + head_off + idx] = o13 * inv1;
  }
}

// Slow path: one block per (head, query row); threads cover the head width and
// the score dot is reduced across the block.  Used when d_k > MAX_DK.
__global__ void mha_general_kernel(const float* __restrict__ Q,
                                   const float* __restrict__ K,
                                   const float* __restrict__ V,
                                   float* __restrict__ output, int N,
                                   int d_model, int h, int d_k, float scale) {
  const int head = blockIdx.x;
  const int q_row = blockIdx.y;
  const int tid = threadIdx.x;
  const int col_offset = head * d_k;

  extern __shared__ float smem[];
  float* s_reduce = smem + d_k;
  float* s_scalar = smem + d_k + 32;

  float q_val = 0.0f;
  float acc = 0.0f;
  if (tid < d_k) q_val = Q[q_row * d_model + col_offset + tid];
  __syncthreads();

  float m_i = -FLT_MAX;
  float l_i = 0.0f;

  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;
  const int num_warps = (blockDim.x + 31) >> 5;

  for (int j = 0; j < N; ++j) {
    const int kv_base = j * d_model + col_offset;
    float partial = (tid < d_k) ? (q_val * K[kv_base + tid]) : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    if (lane_id == 0) s_reduce[warp_id] = partial;
    __syncthreads();

    if (warp_id == 0) {
      float sum = (lane_id < num_warps) ? s_reduce[lane_id] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      if (lane_id == 0) {
        const float score = sum * scale;
        const float m_new = fmaxf(m_i, score);
        const float alpha = expf(m_i - m_new);
        const float beta = expf(score - m_new);
        l_i = l_i * alpha + beta;
        m_i = m_new;
        s_scalar[0] = alpha;
        s_scalar[1] = beta;
      }
    }
    __syncthreads();

    if (tid < d_k) {
      acc = acc * s_scalar[0] + s_scalar[1] * V[kv_base + tid];
    }
    __syncthreads();
  }

  if (warp_id == 0 && lane_id == 0) s_scalar[0] = l_i;
  __syncthreads();
  if (tid < d_k) {
    output[q_row * d_model + col_offset + tid] = acc / s_scalar[0];
  }
}

// Q, K, V, output are device pointers (N x d_model, row-major).
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int N, int d_model, int h) {
  if (Q == nullptr || K == nullptr || V == nullptr || output == nullptr) return;
  if (N <= 0 || d_model <= 0 || h <= 0 || d_model % h != 0) return;

  const int d_k = d_model / h;
  const float scale = 1.0f / sqrtf(static_cast<float>(d_k));

  if (d_k <= MAX_DK) {
    const dim3 grid((N + BLOCK_M - 1) / BLOCK_M, h);
    mha_flash_kernel<<<grid, WARPS_PER_BLOCK * 32>>>(
        Q, K, V, output, N, d_model, d_k, scale);
  } else {
    int threads = ((d_k + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    const dim3 grid(h, N);
    const size_t smem_bytes = static_cast<size_t>(d_k + 34) * sizeof(float);
    mha_general_kernel<<<grid, threads, smem_bytes>>>(
        Q, K, V, output, N, d_model, h, d_k, scale);
  }
  cudaDeviceSynchronize();
}
