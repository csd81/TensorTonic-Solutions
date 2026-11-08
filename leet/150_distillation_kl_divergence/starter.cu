#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

struct OnlineState {
    float m_a; // Running max of scaled teacher logits
    float d_a; // Running sum of exp(a - m_a)
    float n_a; // Running sum of exp(a - m_a) * (a - b)
    float m_b; // Running max of scaled student logits
    float d_b; // Running sum of exp(b - m_b)
};

__device__ __forceinline__ OnlineState combine_states(OnlineState s1, OnlineState s2) {
    if (s1.d_a == 0.0f) return s2;
    if (s2.d_a == 0.0f) return s1;

    OnlineState res;

    // Combine teacher (a) and cross-term states
    if (s1.m_a >= s2.m_a) {
        float scale = expf(s2.m_a - s1.m_a);
        res.m_a = s1.m_a;
        res.d_a = s1.d_a + s2.d_a * scale;
        res.n_a = s1.n_a + s2.n_a * scale;
    } else {
        float scale = expf(s1.m_a - s2.m_a);
        res.m_a = s2.m_a;
        res.d_a = s1.d_a * scale + s2.d_a;
        res.n_a = s1.n_a * scale + s2.n_a;
    }

    // Combine student (b) states
    if (s1.m_b >= s2.m_b) {
        float scale = expf(s2.m_b - s1.m_b);
        res.m_b = s1.m_b;
        res.d_b = s1.d_b + s2.d_b * scale;
    } else {
        float scale = expf(s1.m_b - s2.m_b);
        res.m_b = s2.m_b;
        res.d_b = s1.d_b * scale + s2.d_b;
    }

    return res;
}

__device__ __forceinline__ OnlineState update_state(OnlineState s, float a, float b) {
    float diff = a - b;

    // Update teacher (a) and weighted difference accumulator
    if (a > s.m_a) {
        float scale = (s.d_a == 0.0f) ? 0.0f : expf(s.m_a - a);
        s.m_a = a;
        s.d_a = s.d_a * scale + 1.0f;
        s.n_a = s.n_a * scale + diff;
    } else {
        float w = expf(a - s.m_a);
        s.d_a += w;
        s.n_a += w * diff;
    }

    // Update student (b)
    if (b > s.m_b) {
        float scale = (s.d_b == 0.0f) ? 0.0f : expf(s.m_b - b);
        s.m_b = b;
        s.d_b = s.d_b * scale + 1.0f;
    } else {
        s.d_b += expf(b - s.m_b);
    }

    return s;
}

__device__ __forceinline__ OnlineState warp_reduce(OnlineState val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        OnlineState other;
        other.m_a = __shfl_down_sync(0xffffffff, val.m_a, offset);
        other.d_a = __shfl_down_sync(0xffffffff, val.d_a, offset);
        other.n_a = __shfl_down_sync(0xffffffff, val.n_a, offset);
        other.m_b = __shfl_down_sync(0xffffffff, val.m_b, offset);
        other.d_b = __shfl_down_sync(0xffffffff, val.d_b, offset);
        val = combine_states(val, other);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void kl_divergence_kernel(
    const float* __restrict__ teacher_logits,
    const float* __restrict__ student_logits,
    float* __restrict__ kl,
    int V,
    float inv_T) 
{
    int row = blockIdx.x;
    const float* t_row = teacher_logits + (size_t)row * V;
    const float* s_row = student_logits + (size_t)row * V;

    OnlineState state = {-FLT_MAX, 0.0f, 0.0f, -FLT_MAX, 0.0f};

    // Check 16-byte alignment for vectorized float4 loads
    bool aligned = (((uintptr_t)t_row & 15) == 0) && (((uintptr_t)s_row & 15) == 0);
    int j = threadIdx.x;

    if (aligned) {
        int V4 = V / 4;
        const float4* t4 = reinterpret_cast<const float4*>(t_row);
        const float4* s4 = reinterpret_cast<const float4*>(s_row);

        for (; j < V4; j += BLOCK_SIZE) {
            float4 tv = t4[j];
            float4 sv = s4[j];
            state = update_state(state, tv.x * inv_T, sv.x * inv_T);
            state = update_state(state, tv.y * inv_T, sv.y * inv_T);
            state = update_state(state, tv.z * inv_T, sv.z * inv_T);
            state = update_state(state, tv.w * inv_T, sv.w * inv_T);
        }

        // Handle remaining 0..3 elements
        for (int rem = V4 * 4 + threadIdx.x; rem < V; rem += BLOCK_SIZE) {
            state = update_state(state, t_row[rem] * inv_T, s_row[rem] * inv_T);
        }
    } else {
        for (; j < V; j += BLOCK_SIZE) {
            state = update_state(state, t_row[j] * inv_T, s_row[j] * inv_T);
        }
    }

    // Intra-warp reduction
    state = warp_reduce(state);

    // Inter-warp reduction via shared memory
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ OnlineState shared[NUM_WARPS];

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    if (lane == 0) {
        shared[wid] = state;
    }
    __syncthreads();

    if (wid == 0) {
        state = (lane < NUM_WARPS) ? shared[lane] : OnlineState{-FLT_MAX, 0.0f, 0.0f, -FLT_MAX, 0.0f};
        state = warp_reduce(state);

        if (lane == 0) {
            float lse_a = state.m_a + logf(state.d_a);
            float lse_b = state.m_b + logf(state.d_b);
            float expected_diff = state.n_a / state.d_a;
            float res = (lse_b - lse_a) + expected_diff;
            
            // Clamp minor floating-point negative drift near 0
            kl[row] = (res < 0.0f && res > -1e-6f) ? 0.0f : res;
        }
    }
}

// teacher_logits, student_logits, kl are device pointers
extern "C" void solve(const float* teacher_logits, const float* student_logits, float* kl, int N,
                      int V, float temperature) {
    if (N <= 0 || V <= 0) return;

    float inv_T = 1.0f / temperature;
    constexpr int BLOCK_SIZE = 512;

    kl_divergence_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE>>>(
        teacher_logits, student_logits, kl, V, inv_T
    );
}
