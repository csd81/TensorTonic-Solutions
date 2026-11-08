#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ void unit_rmsnorm2(float x0, float x1, float& y0, float& y1) {
    // mean(x^2) = 0.5f * (x0^2 + x1^2)
    float ms = 0.5f * (x0 * x0 + x1 * x1);
    float inv_rms = rsqrtf(ms + 1e-6f);
    y0 = x0 * inv_rms;
    y1 = x1 * inv_rms;
}

__global__ void adder_transformer_kernel(
    const int* __restrict__ prompts,
    float* __restrict__ output,
    const float* __restrict__ weights,
    int batch_size,
    float attn_scale
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    // Load 10 model weights into registers
    const float w0    = weights[0];
    const float w1    = weights[1];
    const float q0    = weights[2];
    const float q1    = weights[3];
    const float v0    = weights[4];
    const float a     = weights[5];
    const float c     = weights[6];
    const float carry = weights[7];
    const float n0    = weights[8];
    const float n1    = weights[9];

    // Precompute embedding table E[d] = [w0 - w1 * d^2, -d]
    float E0[10];
    float E1[10];
    #pragma unroll
    for (int d = 0; d < 10; ++d) {
        float df = static_cast<float>(d);
        E0[d] = w0 - w1 * (df * df);
        E1[d] = -df;
    }

    // Precompute RoPE cos/sin for positions 0..40 (max index needed is 31 + 11 - 1 = 41, index 40)
    const double omega = 2.0 * M_PI / 19.0;
    float cos_p[41];
    float sin_p[41];
    #pragma unroll
    for (int p = 0; p < 41; ++p) {
        double angle = static_cast<double>(p) * omega;
        cos_p[p] = static_cast<float>(cos(angle));
        sin_p[p] = static_cast<float>(sin(angle));
    }

    // KV Cache in thread-local registers (41 positions max)
    float K0_cache[41];
    float K1_cache[41];
    float V0_cache[41];

    const int* prompt_ptr = prompts + static_cast<size_t>(b) * 31;
    float* out_ptr = output + static_cast<size_t>(b) * 110; // 11 * 10

    // Helper lambda to compute K, V at position p for token d, and optionally full forward pass
    auto forward_token = [&](int p, int d, bool compute_logits) -> int {
        // 1. Token Embedding
        float x0 = E0[d];
        float x1 = E1[d];

        // 2. Unit RMSNorm before Attention
        float h0, h1;
        unit_rmsnorm2(x0, x1, h0, h1);

        // 3. Self-Attention Projections
        // K = [h0, 0] -> UnitRMSNorm(K) = [sign(h0) or h0 / sqrt(0.5*h0^2 + 1e-6), 0]
        float k_raw0 = h0;
        float k_norm0, k_norm1;
        unit_rmsnorm2(k_raw0, 0.0f, k_norm0, k_norm1); // k_norm1 is 0.0f

        // Apply RoPE to K
        float cp = cos_p[p];
        float sp = sin_p[p];
        K0_cache[p] = k_norm0 * cp;
        K1_cache[p] = k_norm0 * sp;

        // V = [h1 * v0, 0] (only V[0] is non-zero and used)
        V0_cache[p] = h1 * v0;

        if (!compute_logits) {
            return 0;
        }

        // Q = [h0 * q0, h0 * q1]
        float q_norm0, q_norm1;
        unit_rmsnorm2(h0 * q0, h0 * q1, q_norm0, q_norm1);

        // Apply RoPE to Q
        float qp0 = q_norm0 * cp - q_norm1 * sp;
        float qp1 = q_norm0 * sp + q_norm1 * cp;

        // Causal Dot-Product Attention over j = 0..p (numerically stable softmax)
        float scores[41];
        float max_score = -FLT_MAX;
        for (int j = 0; j <= p; ++j) {
            float s = (qp0 * K0_cache[j] + qp1 * K1_cache[j]) * attn_scale;
            scores[j] = s;
            if (s > max_score) max_score = s;
        }

        float sum_exp = 0.0f;
        float attn0 = 0.0f;
        for (int j = 0; j <= p; ++j) {
            float w = expf(scores[j] - max_score);
            sum_exp += w;
            attn0 += w * V0_cache[j];
        }
        attn0 /= sum_exp;

        // Output projection [attn0, attn1] -> [0, attn0] + Residual
        x1 += attn0;

        // 4. MLP Block
        unit_rmsnorm2(x0, x1, h0, h1);

        float g0 = h0 * a + h1 * c;
        float g1 = h0 * (a - c * 0.001f) + h1 * c;

        float mix0 = silu_f(g0) * h0;
        float mix1 = silu_f(g1) * h0;

        // MLP(h) = [0, carry * (mix1 - mix0)] + Residual
        x1 += carry * (mix1 - mix0);

        // 5. Final RMSNorm with learned weights [n0, n1]
        float fn0, fn1;
        unit_rmsnorm2(x0, x1, fn0, fn1);
        float out0 = fn0 * n0;
        float out1 = fn1 * n1;

        // 6. Output Logits & Argmax
        int best_tok = 0;
        float best_logit = -FLT_MAX;
        float* step_out = out_ptr + (p - 30) * 10;

        #pragma unroll
        for (int tok = 0; tok < 10; ++tok) {
            float logit = out0 * E0[tok] + out1 * E1[tok];
            step_out[tok] = logit;
            if (logit > best_logit) {
                best_logit = logit;
                best_tok = tok;
            }
        }

        return best_tok;
    };

    // Prefill positions 0..29 (only K and V are needed for causal attention at step >= 30)
    for (int p = 0; p < 30; ++p) {
        forward_token(p, prompt_ptr[p], false);
    }

    // Autoregressive decoding: 11 steps (positions 30..40)
    int current_tok = prompt_ptr[30];
    for (int step = 0; step < 11; ++step) {
        int p = 30 + step;
        current_tok = forward_token(p, current_tok, true);
    }
}

extern "C" void solve(const int* prompts, float* output, const float* weights, int batch_size) {
    if (batch_size <= 0) return;

    // Precompute attention scale: (1 / sqrt(2)) * S^2
    // S^2 = ln(10) / (sqrt(2) * (cos(0.3 * omega) - cos(0.7 * omega)))
    // Simplifies to: ln(10) / (2 * (cos(0.3 * omega) - cos(0.7 * omega)))
    const double omega = 2.0 * M_PI / 19.0;
    const double s_sq = log(10.0) / (sqrt(2.0) * (cos(0.3 * omega) - cos(0.7 * omega)));
    const float attn_scale = static_cast<float>((1.0 / sqrt(2.0)) * s_sq);

    const int threads_per_block = 256;
    const int blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    adder_transformer_kernel<<<blocks, threads_per_block>>>(
        prompts,
        output,
        weights,
        batch_size,
        attn_scale
    );
}
