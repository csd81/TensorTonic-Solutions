// Naive baseline for challenge 38 (nearest neighbor).
//
// The plainest correct form, kept close to the starter: one query per thread,
// points staged in an interleaved (AoS) shared-memory tile, self-index test
// inside the inner loop. No SoA transpose, no query register blocking, no
// fast path.

#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 256

__global__ void nearest_neighbor_naive_kernel(const float* __restrict__ points,
                                              int* __restrict__ indices, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load query point i into registers
    float qx = 0.0f, qy = 0.0f, qz = 0.0f;
    if (i < N) {
        qx = points[3 * i];
        qy = points[3 * i + 1];
        qz = points[3 * i + 2];
    }

    float min_dist = FLT_MAX;
    int best_idx = -1;

    __shared__ float tile[TILE_SIZE * 3];

    for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
        int load_idx = tile_start + threadIdx.x;
        if (load_idx < N) {
            tile[3 * threadIdx.x]     = points[3 * load_idx];
            tile[3 * threadIdx.x + 1] = points[3 * load_idx + 1];
            tile[3 * threadIdx.x + 2] = points[3 * load_idx + 2];
        }
        __syncthreads();

        if (i < N) {
            int tile_end = min(TILE_SIZE, N - tile_start);
            for (int k = 0; k < tile_end; ++k) {
                int j = tile_start + k;
                if (j != i) {
                    float dx = qx - tile[3 * k];
                    float dy = qy - tile[3 * k + 1];
                    float dz = qz - tile[3 * k + 2];
                    float dist_sq = dx * dx + dy * dy + dz * dz;
                    if (dist_sq < min_dist) {
                        min_dist = dist_sq;
                        best_idx = j;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (i < N) {
        indices[i] = best_idx;
    }
}

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {
    if (N <= 0) return;
    int threads_per_block = TILE_SIZE;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    nearest_neighbor_naive_kernel<<<blocks_per_grid, threads_per_block>>>(
        points, indices, N);
}
