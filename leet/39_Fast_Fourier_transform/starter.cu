#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Complex arithmetic helpers using float2 (x = real, y = imag)
__device__ __forceinline__ float2 c_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 c_sub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 c_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 c_conj(float2 a) {
    return make_float2(a.x, -a.y);
}

// Bit-reversal permutation for power-of-2 FFT
__global__ void bit_reverse_kernel(const float2* __restrict__ in, float2* __restrict__ out, int M, int log2M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        unsigned int rev = __brev(idx) >> (32 - log2M);
        out[rev] = in[idx];
    }
}

// Single stage of iterative Cooley-Tukey Radix-2 FFT
__global__ void fft_stage_kernel(float2* __restrict__ data, int M, int step, bool inverse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_step = step >> 1;
    int total_butterflies = M >> 1;

    if (idx < total_butterflies) {
        int group = idx / half_step;
        int j = idx % half_step;
        int even_idx = group * step + j;
        int odd_idx = even_idx + half_step;

        // Compute twiddle factor in double precision for accuracy
        double angle = (inverse ? 2.0 : -2.0) * M_PI * (double)j / (double)step;
        float2 w = make_float2((float)cos(angle), (float)sin(angle));

        float2 u = data[even_idx];
        float2 v = c_mul(w, data[odd_idx]);

        data[even_idx] = c_add(u, v);
        data[odd_idx] = c_sub(u, v);
    }
}

// Scale array by 1/M after inverse FFT
__global__ void scale_kernel(float2* __restrict__ data, int M, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

// Step 1 of Bluestein: Prepare padded input array A and chirp kernel B
__global__ void bluestein_prep_kernel(const float2* __restrict__ signal, float2* __restrict__ A, float2* __restrict__ B, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        if (idx < N) {
            // Use modulo 2N to prevent precision loss for large idx^2
            long long idx_sq = ((long long)idx * (long long)idx) % (2LL * N);
            double angle = M_PI * (double)idx_sq / (double)N;
            float2 chirp = make_float2((float)cos(angle), (float)sin(angle));

            // a[n] = x[n] * exp(-j * pi * n^2 / N)
            A[idx] = c_mul(signal[idx], c_conj(chirp));
            // b[n] = exp(+j * pi * n^2 / N)
            B[idx] = chirp;
        } else {
            A[idx] = make_float2(0.0f, 0.0f);
            if (idx > M - N) {
                int n = M - idx;
                long long n_sq = ((long long)n * (long long)n) % (2LL * N);
                double angle = M_PI * (double)n_sq / (double)N;
                // Wrap around negative indices: b[M - n] = b[n]
                B[idx] = make_float2((float)cos(angle), (float)sin(angle));
            } else {
                B[idx] = make_float2(0.0f, 0.0f);
            }
        }
    }
}

// Pointwise complex multiplication in frequency domain
__global__ void pointwise_mul_kernel(float2* __restrict__ A, const float2* __restrict__ B, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        A[idx] = c_mul(A[idx], B[idx]);
    }
}

// Final step of Bluestein: Multiply convolved result by chirp to obtain X[k]
__global__ void bluestein_post_kernel(const float2* __restrict__ A, float2* __restrict__ spectrum, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        long long k_sq = ((long long)k * (long long)k) % (2LL * N);
        double angle = -M_PI * (double)k_sq / (double)N;
        float2 chirp_neg = make_float2((float)cos(angle), (float)sin(angle));
        spectrum[k] = c_mul(A[k], chirp_neg);
    }
}

// Helper to execute an in-place power-of-2 FFT using a scratch buffer
void run_pow2_fft(float2* d_data, float2* d_scratch, int M, int log2M, bool inverse) {
    int threads = 256;
    int blocks_M = (M + threads - 1) / threads;
    int blocks_half = ((M >> 1) + threads - 1) / threads;

    bit_reverse_kernel<<<blocks_M, threads>>>(d_data, d_scratch, M, log2M);
    cudaMemcpy(d_data, d_scratch, M * sizeof(float2), cudaMemcpyDeviceToDevice);

    for (int step = 2; step <= M; step <<= 1) {
        fft_stage_kernel<<<blocks_half, threads>>>(d_data, M, step, inverse);
    }

    if (inverse) {
        scale_kernel<<<blocks_M, threads>>>(d_data, M, 1.0f / (float)M);
    }
}

// signal and spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int N) {
    if (N <= 0) return;

    const float2* d_sig = reinterpret_cast<const float2*>(signal);
    float2* d_spec = reinterpret_cast<float2*>(spectrum);

    if (N == 1) {
        cudaMemcpy(d_spec, d_sig, sizeof(float2), cudaMemcpyDeviceToDevice);
        return;
    }

    int threads = 256;

    // Fast Path: N is already a power of 2
    if ((N & (N - 1)) == 0) {
        int log2N = 0;
        while ((1 << log2N) < N) log2N++;

        float2* d_scratch;
        cudaMalloc(&d_scratch, N * sizeof(float2));

        int blocks = (N + threads - 1) / threads;
        bit_reverse_kernel<<<blocks, threads>>>(d_sig, d_spec, N, log2N);

        int blocks_half = ((N >> 1) + threads - 1) / threads;
        for (int step = 2; step <= N; step <<= 1) {
            fft_stage_kernel<<<blocks_half, threads>>>(d_spec, N, step, false);
        }

        cudaFree(d_scratch);
        return;
    }

    // General Path: Bluestein's Algorithm for arbitrary N
    // Find next power of 2 >= 2N - 1
    int M = 1;
    int log2M = 0;
    while (M < 2 * N - 1) {
        M <<= 1;
        log2M++;
    }

    float2 *d_A, *d_B, *d_scratch;
    cudaMalloc(&d_A, M * sizeof(float2));
    cudaMalloc(&d_B, M * sizeof(float2));
    cudaMalloc(&d_scratch, M * sizeof(float2));

    int blocks_M = (M + threads - 1) / threads;
    int blocks_N = (N + threads - 1) / threads;

    // 1. Initialize modulated input A and chirp filter B
    bluestein_prep_kernel<<<blocks_M, threads>>>(d_sig, d_A, d_B, N, M);

    // 2. Transform both A and B to frequency domain
    run_pow2_fft(d_A, d_scratch, M, log2M, false);
    run_pow2_fft(d_B, d_scratch, M, log2M, false);

    // 3. Convolve via pointwise multiplication and inverse FFT
    pointwise_mul_kernel<<<blocks_M, threads>>>(d_A, d_B, M);
    run_pow2_fft(d_A, d_scratch, M, log2M, true);

    // 4. Demodulate final output into spectrum
    bluestein_post_kernel<<<blocks_N, threads>>>(d_A, d_spec, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_scratch);
}
