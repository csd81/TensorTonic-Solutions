#include <cuda_runtime.h>
#include <cmath>

#define PI 3.14159265358979323846f

// Computes 1D DFT along each row: x[m, n] -> intermediate[m, l]
__global__ void dft_rows(const float* __restrict__ signal, float* __restrict__ intermediate, int M, int N) {
    // Each block handles one row
    int m = blockIdx.x;
    if (m >= M) return;

    // Threads in the block cooperatively compute the columns (l) of the output spectrum
    for (int l = threadIdx.x; l < N; l += blockDim.x) {
        float sum_real = 0.0f;
        float sum_imag = 0.0f;

        // Compute 1D DFT for row m at frequency bin l
        for (int n = 0; n < N; ++n) {
            int signal_idx = 2 * (m * N + n);
            float x_real = signal[signal_idx];
            float x_imag = signal[signal_idx + 1];

            float angle = -2.0f * PI * l * n / (float)N;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            // Complex multiplication: (x_real + i*x_imag) * (cos_a + i*sin_a)
            sum_real += x_real * cos_a - x_imag * sin_a;
            sum_imag += x_real * sin_a + x_imag * cos_a;
        }

        int out_idx = 2 * (m * N + l);
        intermediate[out_idx] = sum_real;
        intermediate[out_idx + 1] = sum_imag;
    }
}

// Computes 1D DFT along each column of the intermediate array: intermediate[m, l] -> spectrum[k, l]
__global__ void dft_cols(const float* __restrict__ intermediate, float* __restrict__ spectrum, int M, int N) {
    // Each block handles one column
    int l = blockIdx.x;
    if (l >= N) return;

    // Threads in the block cooperatively compute the rows (k) of the final spectrum
    for (int k = threadIdx.x; k < M; k += blockDim.x) {
        float sum_real = 0.0f;
        float sum_imag = 0.0f;

        // Compute 1D DFT for column l at frequency bin k
        for (int m = 0; m < M; ++m) {
            int inter_idx = 2 * (m * N + l);
            float inter_real = intermediate[inter_idx];
            float inter_imag = intermediate[inter_idx + 1];

            float angle = -2.0f * PI * k * m / (float)M;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            // Complex multiplication
            sum_real += inter_real * cos_a - inter_imag * sin_a;
            sum_imag += inter_real * sin_a + inter_imag * cos_a;
        }

        int out_idx = 2 * (k * N + l);
        spectrum[out_idx] = sum_real;
        spectrum[out_idx + 1] = sum_imag;
    }
}

extern "C" void solve(const float* signal, float* spectrum, int M, int N) {
    // Allocate temporary device buffer for the intermediate row-transformed results
    float* d_intermediate = nullptr;
    size_t size = M * N * 2 * sizeof(float);
    cudaMalloc(&d_intermediate, size);

    // Execution configuration
    int threadsPerBlock = 256;
    
    // Phase 1: Launch 1D DFT on all rows
    int blocksPerGridRows = M;
    dft_rows<<<blocksPerGridRows, threadsPerBlock>>>(signal, d_intermediate, M, N);

    // Phase 2: Launch 1D DFT on all columns
    int blocksPerGridCols = N;
    dft_cols<<<blocksPerGridCols, threadsPerBlock>>>(d_intermediate, spectrum, M, N);

    // Synchronize to ensure calculations complete before freeing buffer
    cudaDeviceSynchronize();
    cudaFree(d_intermediate);
}
