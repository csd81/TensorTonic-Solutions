#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define EPSILON_SQ (1e-3f * 1e-3f)

__global__ void nbody_kernel(const float* __restrict__ positions, const float* __restrict__ masses, float* __restrict__ forces, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory caches for positions and masses of the current tile
    __shared__ float32_4 shared_pos_mass[BLOCK_SIZE];

    float32_3 force_i = {0.0f, 0.0f, 0.0f};
    float32_3 pos_i = {0.0f, 0.0f, 0.0f};

    if (i < N) {
        pos_i.x = positions[i * 3 + 0];
        pos_i.y = positions[i * 3 + 1];
        pos_i.z = positions[i * 3 + 2];
    }

    // Loop over all bodies in chunks of BLOCK_SIZE
    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        int idx = tile * blockDim.x + threadIdx.x;
        
        // Cooperatively load data into shared memory
        if (idx < N) {
            shared_pos_mass[threadIdx.x].x = positions[idx * 3 + 0];
            shared_pos_mass[threadIdx.x].y = positions[idx * 3 + 1];
            shared_pos_mass[threadIdx.x].z = positions[idx * 3 + 2];
            shared_pos_mass[threadIdx.x].w = masses[idx];
        } else {
            shared_pos_mass[threadIdx.x] = {0.0f, 0.0f, 0.0f, 0.0f};
        }

        __syncthreads();

        // Accumulate forces from the current tile
        if (i < N) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                int global_j = tile * BLOCK_SIZE + j;
                if (global_j >= N) break;

                // Skip self-interaction (force is 0 when i == j)
                if (i != global_j) {
                    float dx = shared_pos_mass[j].x - pos_i.x;
                    float dy = shared_pos_mass[j].y - pos_i.y;
                    float dz = shared_pos_mass[j].z - pos_i.z;
                    float m_j = shared_pos_mass[j].w;

                    float dist_sq = dx * dx + dy * dy + dz * dz + EPSILON_SQ;
                    float inv_dist = rsqrtf(dist_sq);
                    float inv_dist_cubed = inv_dist * inv_dist * inv_dist;

                    float s = m_j * inv_dist_cubed;
                    force_i.x += dx * s;
                    force_i.y += dy * s;
                    force_i.z += dz * s;
                }
            }
        }
        
        __syncthreads();
    }

    // Write computed accumulated forces back to global memory
    if (i < N) {
        forces[i * 3 + 0] = force_i.x;
        forces[i * 3 + 1] = force_i.y;
        forces[i * 3 + 2] = force_i.z;
    }
}

extern "C" void solve(const float* positions, const float* masses, float* forces, int N) {
    if (N <= 0) return;

    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    nbody_kernel<<<blocks_per_grid, threads_per_block>>>(positions, masses, forces, N);
    
    // Ensure kernel completion if required by the host environment
    cudaDeviceSynchronize();
}
