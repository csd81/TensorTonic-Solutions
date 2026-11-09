#include <cuda_runtime.h>
#include <math.h>

// Naive baseline for challenge 103, kept as the starter's shape: one block
// per row, 512 threads, and an integer division + modulo per butterfly to
// recover (section, offset) from the flat pair index.  The shared accesses of
// the small-h stages are bank-conflicted.  Measures 10.06 ms at B=8192,
// N=4096 (26.7 GB/s; the copy floor is 3.29 ms).

// Shared memory kernel to compute the Fast Walsh-Hadamard Transform per row
__global__ void fwhd_naive_kernel(const float* __restrict__ input, float* __restrict__ output, int B, int N) {
    // Dynamically allocated shared memory for storing the row
    extern __shared__ float s_data[];

    int row = blockIdx.x;
    if (row >= B) return;

    int tid = threadIdx.x;
    int b_dim = blockDim.x;

    // 1. Cooperatively load row data from global memory into shared memory
    for (int i = tid; i < N; i += b_dim) {
        s_data[i] = input[row * N + i];
    }
    __syncthreads();

    // 2. Perform the Fast Walsh-Hadamard Transform butterfly stages
    // h represents the stride/distance between paired elements in each stage
    for (int h = 1; h < N; h <<= 1) {
        // Each thread processes elements in a strided loop
        // multiplying h by 2 because each chunk of size 2*h contains pairs
        for (int i = tid; i < (N >> 1); i += b_dim) {
            // Map the flat index i to the actual butterfly pair indices (idx_a, idx_b)
            int section = i / h;
            int offset = i % h;
            int idx_a = section * (2 * h) + offset;
            int idx_b = idx_a + h;

            float a = s_data[idx_a];
            float b = s_data[idx_b];

            s_data[idx_a] = a + b;
            s_data[idx_b] = a - b;
        }
        __syncthreads();
    }

    // 3. Compute the orthonormal scaling factor 1 / sqrt(N)
    float scale = 1.0f / sqrtf((float)N);

    // 4. Cooperatively scale and write the final result back to global memory
    for (int i = tid; i < N; i += b_dim) {
        output[row * N + i] = s_data[i] * scale;
    }
}

extern "C" void solve(const float* input, float* output, int B, int N) {
    // Safeguard for empty edge case executions
    if (B <= 0 || N <= 0) return;

    // Use 512 threads per block as a balanced sweep across N (up to 4096)
    int threadsPerBlock = 512;
    if (threadsPerBlock > N) {
        threadsPerBlock = N;
    }

    // Allocate dynamic shared memory space matching the row size (N floats)
    size_t sharedMemSize = N * sizeof(float);

    // Launch 1 block per independent batch row
    fwhd_naive_kernel<<<B, threadsPerBlock, sharedMemSize>>>(input, output, B, N);
    
    // Ensure accurate and synchronous execution completions
    cudaDeviceSynchronize();
}
