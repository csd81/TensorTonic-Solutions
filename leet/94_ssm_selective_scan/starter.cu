#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel to perform the selective scan forward pass
__global__ void selective_scan_kernel(
    const float* __restrict__ u,     // (batch, seq_len, d_model)
    const float* __restrict__ delta, // (batch, seq_len, d_model)
    const float* __restrict__ A,     // (d_model, d_state)
    const float* __restrict__ B,     // (batch, seq_len, d_state)
    const float* __restrict__ C,     // (batch, seq_len, d_state)
    const float* __restrict__ skip,  // (d_model)
    float* __restrict__ y,           // (batch, seq_len, d_model)
    int batch, int seq_len, int d_model, int d_state) 
{
    // Map each thread to a unique (batch, d_model) channel pair
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch || d >= d_model) return;

    // Dynamically allocate local storage for the hidden state vector (max d_state is 64)
    // Using a fixed size helps the compiler optimize loop structures via register allocation
    float h[64];
    for (int n = 0; n < d_state; ++n) {
        h[n] = 0.0f;
    }

    // Cache the skip connection weight for this channel
    float skip_val = skip[d];

    // Compute base offsets to reduce indexing math inside the inner loop
    int base_u_delta_y = b * seq_len * d_model + d;
    int base_B_C       = b * seq_len * d_state;
    int base_A         = d * d_state;

    // Sequentially scan through the sequence length dimension
    for (int t = 0; t < seq_len; ++t) {
        int idx_u_delta_y = base_u_delta_y + t * d_model;
        int idx_B_C       = base_B_C       + t * d_state;

        float u_val     = u[idx_u_delta_y];
        float dt_val    = delta[idx_u_delta_y];
        float y_acc     = 0.0f;

        for (int n = 0; n < d_state; ++n) {
            float A_val = A[base_A + n];
            float B_val = B[idx_B_C + n];
            float C_val = C[idx_B_C + n];

            // Compute discretized transition parameters A_bar and B_bar
            float A_bar = __expf(dt_val * A_val);
            float B_bar = dt_val * B_val;

            // Recurrently update state and accumulate output
            h[n] = A_bar * h[n] + B_bar * u_val;
            y_acc += C_val * h[n];
        }

        // Apply skip connection and write final result out to global memory
        y[idx_u_delta_y] = y_acc + skip_val * u_val;
    }
}

extern "C" void solve(const float* u, const float* delta, const float* A, const float* B,
                      const float* C, const float* skip, float* y, int batch, int seq_len,
                      int d_model, int d_state) 
{
    // Define a 2D block size optimized for typical warp dimensions (32 x 8 = 256 threads)
    dim3 threadsPerBlock(32, 8);
    
    // Calculate grid dimensions ensuring coverage for all batches and model channels
    dim3 numBlocks(
        (d_model + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (batch   + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch the scan kernel execution on the current stream
    selective_scan_kernel<<<numBlocks, threadsPerBlock>>>(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state
    );
}
