#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 256
#define TILE_DIM 32
#define BLOCK_ROWS 8

// 1. Performs an inclusive scan within chunks of size BLOCK_SIZE along each row
// and stores the total sum of each chunk into block_sums.
__global__ void row_scan_chunks_kernel(const float* __restrict__ in, 
                                       float* __restrict__ out, 
                                       float* __restrict__ block_sums, 
                                       int H, int W, int blocks_per_row) {
    __shared__ float temp[BLOCK_SIZE];
    
    int row = blockIdx.y;
    int chunk_idx = blockIdx.x;
    int tid = threadIdx.x;
    int col = chunk_idx * BLOCK_SIZE + tid;
    
    if (row >= H) return;

    float val = (col < W) ? in[row * W + col] : 0.0f;
    temp[tid] = val;
    __syncthreads();

    // Inclusive Hillis-Steele scan in shared memory
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        float t = 0.0f;
        if (tid >= offset) {
            t = temp[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            temp[tid] += t;
        }
        __syncthreads();
    }

    if (col < W) {
        out[row * W + col] = temp[tid];
    }

    // Last thread in the block writes the block's total sum
    if (tid == BLOCK_SIZE - 1 && block_sums != nullptr) {
        block_sums[row * blocks_per_row + chunk_idx] = temp[tid];
    }
}

// 2. Scans the block_sums array for each row (supports up to 16,384 / 256 = 64 blocks per row)
__global__ void scan_block_sums_kernel(float* __restrict__ block_sums, int H, int blocks_per_row) {
    __shared__ float temp[64]; // Max blocks_per_row for W <= 16384 with BLOCK_SIZE=256 is 64
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= H || tid >= blocks_per_row) return;
    
    temp[tid] = block_sums[row * blocks_per_row + tid];
    __syncthreads();
    
    for (int offset = 1; offset < blocks_per_row; offset *= 2) {
        float t = 0.0f;
        if (tid >= offset) {
            t = temp[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            temp[tid] += t;
        }
        __syncthreads();
    }
    
    block_sums[row * blocks_per_row + tid] = temp[tid];
}

// 3. Adds the scanned block offsets back to each chunk
__global__ void add_block_offsets_kernel(float* __restrict__ data, 
                                         const float* __restrict__ block_sums, 
                                         int H, int W, int blocks_per_row) {
    int row = blockIdx.y;
    int chunk_idx = blockIdx.x;
    int tid = threadIdx.x;
    int col = chunk_idx * BLOCK_SIZE + tid;
    
    if (row >= H || col >= W || chunk_idx == 0) return;
    
    float offset = block_sums[row * blocks_per_row + (chunk_idx - 1)];
    data[row * W + col] += offset;
}

// 4. Coalesced tiled matrix transpose with padding to prevent shared memory bank conflicts
__global__ void transpose_kernel(const float* __restrict__ in, float* __restrict__ out, int H, int W) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < W && (y + j) < H) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * W + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < H && (y + j) < W) {
            out[(y + j) * H + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Helper to execute a full 1D inclusive scan across all rows of a matrix
void scan_all_rows(const float* d_in, float* d_out, float* d_block_sums, int rows, int cols) {
    int blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_chunks(blocks_per_row, rows);
    
    row_scan_chunks_kernel<<<grid_chunks, BLOCK_SIZE>>>(
        d_in, d_out, (blocks_per_row > 1) ? d_block_sums : nullptr, rows, cols, blocks_per_row
    );

    if (blocks_per_row > 1) {
        scan_block_sums_kernel<<<rows, 64>>>(d_block_sums, rows, blocks_per_row);
        add_block_offsets_kernel<<<grid_chunks, BLOCK_SIZE>>>(d_out, d_block_sums, rows, cols, blocks_per_row);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int H, int W) {
    size_t num_elements = static_cast<size_t>(H) * W;
    size_t bytes = num_elements * sizeof(float);

    int max_dim = (H > W) ? H : W;
    int max_blocks_per_row = (max_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate temporary buffers on device
    float* d_temp = nullptr;
    float* d_block_sums = nullptr;
    cudaMalloc(&d_temp, bytes);
    cudaMalloc(&d_block_sums, static_cast<size_t>(max_dim) * max_blocks_per_row * sizeof(float));

    // Step 1: Horizontal inclusive prefix sum (input -> output)
    scan_all_rows(input, output, d_block_sums, H, W);

    // Step 2: Transpose (output [H x W] -> d_temp [W x H])
    dim3 grid_trans1((W + TILE_DIM - 1) / TILE_DIM, (H + TILE_DIM - 1) / TILE_DIM);
    dim3 block_trans(TILE_DIM, BLOCK_ROWS);
    transpose_kernel<<<grid_trans1, block_trans>>>(output, d_temp, H, W);

    // Step 3: Vertical inclusive prefix sum via row scan on transposed data (d_temp -> d_temp)
    scan_all_rows(d_temp, d_temp, d_block_sums, W, H);

    // Step 4: Transpose back to row-major layout (d_temp [W x H] -> output [H x W])
    dim3 grid_trans2((H + TILE_DIM - 1) / TILE_DIM, (W + TILE_DIM - 1) / TILE_DIM);
    transpose_kernel<<<grid_trans2, block_trans>>>(d_temp, output, W, H);

    // Free temporary device memory
    cudaFree(d_temp);
    cudaFree(d_block_sums);
}
