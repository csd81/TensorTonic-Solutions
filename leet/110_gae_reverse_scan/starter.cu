
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float warp_reverse_scan(float val, float decay) {
    unsigned mask = 0xffffffff;
    int lane = threadIdx.x & 31;
    float d = decay;

    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_down_sync(mask, val, offset);
        if (lane + offset < 32) {
            val += d * n;
        }
        d *= d;
    }
    return val;
}

__global__ void gae_reverse_scan_kernel(
    const float* __restrict__ rewards,
    const float* __restrict__ values,
    float* __restrict__ advantages,
    float gamma,
    float c,
    int S
) {
    int b = blockIdx.x;
    const float* r_row = rewards + b * S;
    const float* v_row = values + b * S;
    float* adv_row = advantages + b * S;

    int tid = threadIdx.x;          // 0 .. 1023
    int lane = tid & 31;            // 0 .. 31
    int warp_id = tid >> 5;         // 0 .. 31
    const int NUM_WARPS = 32;
    const int CHUNK_SIZE = 1024;

    __shared__ float warp_totals[NUM_WARPS];
    __shared__ float chunk_carry;
    __shared__ float prev_chunk_carry;

    if (tid == 0) {
        chunk_carry = 0.0f;
    }
    __syncthreads();

    float c32 = powf(c, 32.0f);

    int num_chunks = (S + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for (int chunk = num_chunks - 1; chunk >= 0; --chunk) {
        int t = chunk * CHUNK_SIZE + tid;

        // 1. Compute one-step TD error (0.0f for out-of-bounds threads so padding decays cleanly)
        float delta = 0.0f;
        if (t < S) {
            float v_curr = v_row[t];
            float v_next = (t + 1 < S) ? v_row[t + 1] : 0.0f;
            delta = r_row[t] + gamma * v_next - v_curr;
        }

        // 2. Intra-warp reverse scan
        float warp_val = warp_reverse_scan(delta, c);

        if (lane == 0) {
            warp_totals[warp_id] = warp_val;
        }
        __syncthreads();

        // 3. Inter-warp reverse scan in Warp 0
        if (warp_id == 0) {
            if (lane == 0) {
                prev_chunk_carry = chunk_carry;
            }
            float w_val = warp_totals[lane];
            float w_scanned = warp_reverse_scan(w_val, c32);

            // Add decayed carry coming from the previous (right-hand) 1024-element chunk
            w_scanned += prev_chunk_carry * powf(c32, (float)(31 - lane));
            warp_totals[lane] = w_scanned;

            if (lane == 0) {
                chunk_carry = w_scanned;
            }
        }
        __syncthreads();

        // 4. Combine intra-warp result with carry from the right-hand warp/chunk
        if (t < S) {
            float right_carry = (warp_id + 1 < NUM_WARPS) ? warp_totals[warp_id + 1] : prev_chunk_carry;
            float adv = warp_val + right_carry * powf(c, (float)(32 - lane));
            adv_row[t] = adv;
        }
        __syncthreads();
    }
}

// rewards, values, advantages are device pointers
extern "C" void solve(const float* rewards, const float* values, float* advantages, float gamma,
                      float lam, int B, int S) {
    if (B <= 0 || S <= 0) return;

    float c = gamma * lam;
    int threads_per_block = 1024;
    gae_reverse_scan_kernel<<<B, threads_per_block>>>(rewards, values, advantages, gamma, c, S);
}
