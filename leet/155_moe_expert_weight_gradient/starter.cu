#include <cuda_runtime.h>

// Tile dimensions
constexpr int BM = 128; // Block tile size along K
constexpr int BN = 128; // Block tile size along N
constexpr int BK = 16;  // Block tile size along M_total (reduction dim)
constexpr int TM = 8;   // Thread tile size along K
constexpr int TN = 8;   // Thread tile size along N

__global__ void moe_weight_grad_kernel(
    const float* __restrict__ X,
    const float* __restrict__ dY,
    const int* __restrict__ group_offsets,
    float* __restrict__ dW,
    int G, int K, int N) {

    // Identify expert g and output tile coordinates (row in K, col in N)
    const int g = blockIdx.z;
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;

    // Fetch token boundaries [s, e) for expert g
    const int s = group_offsets[g];
    const int e = group_offsets[g + 1];

    // Thread coordinates within the 16x16 thread block (256 threads)
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int ty = threadIdx.y; // [0, 15]
    const int tx = threadIdx.x; // [0, 15]

    // Accumulator registers initialized to 0.0f
    // Automatically handles empty experts (s == e) by writing zeros at the end
    float acc[TM][TN] = {0.0f};

    // Shared memory buffers with +1 padding on the inner dimension to avoid bank conflicts
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];

    // Indices for loading 256 elements (BK * BM = 16 * 128 = 2048 -> 8 elements per thread)
    const int load_m = tid / 32;        // [0, 7] -> step of 8 along BK
    const int load_k = (tid % 32) * 4;  // [0, 124] -> 4 floats per thread

    // Loop over the ragged reduction dimension m in chunks of BK
    for (int m_tile = s; m_tile < e; m_tile += BK) {
        // Cooperatively load X[m_tile : m_tile+BK, row_base : row_base+BM] into As (transposed)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int m_off = load_m + i * 8;
            int m_idx = m_tile + m_off;
            
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                int k_off = load_k + v;
                int k_idx = row_base + k_off;
                if (m_idx < e && k_idx < K) {
                    As[m_off][k_off] = X[m_idx * K + k_idx];
                } else {
                    As[m_off][k_off] = 0.0f;
                }
            }
        }

        // Cooperatively load dY[m_tile : m_tile+BK, col_base : col_base+BN] into Bs
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int m_off = load_m + i * 8;
            int m_idx = m_tile + m_off;

            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                int n_off = load_k + v; // load_k covers, same as BN
                int n_idx = col_base + n_off;
                if (m_idx < e && n_idx < N) {
                    Bs[m_off][n_off] = dY[m_idx * N + n_idx];
                } else {
                    Bs[m_off][n_off] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute outer products across the BK tile in registers
        #pragma unroll
        for (int m = 0; m < BK; ++m) {
            float reg_a[TM];
            float reg_b[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[m][ty * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs[m][tx * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write the 8x8 register tile back to global memory dW[g]
    float* dW_g = dW + static_cast<size_t>(g) * K * N;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int out_row = row_base + ty * TM + i;
        if (out_row < K) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int out_col = col_base + tx * TN + j;
                if (out_col < N) {
                    dW_g[out_row * N + out_col] = acc[i][j];
                }
            }
        }
    }
}

// X, dY, group_offsets, dW are device pointers
extern "C" void solve(const float* X, const float* dY, const int* group_offsets, float* dW, int G,
                      int M_total, int K, int N) {
    dim3 block(16, 16); // 256 threads per block
    dim3 grid(
        (N + BN - 1) / BN,
        (K + BM - 1) / BM,
        G
    );

    moe_weight_grad_kernel<<<grid, block>>>(X, dY, group_offsets, dW, G, K, N);
}
