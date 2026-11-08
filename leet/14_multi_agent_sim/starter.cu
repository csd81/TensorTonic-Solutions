#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void boids_kernel(const float4* __restrict__ agents, float4* __restrict__ agents_next, int N) {
    __shared__ float4 tile[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load current agent's state into registers if within bounds
    float4 my_agent = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i < N) {
        my_agent = agents[i];
    }

    float sum_vx = 0.0f;
    float sum_vy = 0.0f;
    int neighbor_count = 0;

    // Radius r = 5.0f -> r^2 = 25.0f (avoids costly sqrtf calls)
    const float r_sq = 25.0f;
    const float alpha = 0.05f;

    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Iterate over all agents in shared memory tiles
    for (int t = 0; t < num_tiles; ++t) {
        int j_load = t * BLOCK_SIZE + threadIdx.x;
        if (j_load < N) {
            tile[threadIdx.x] = agents[j_load];
        }
        __syncthreads();

        if (i < N) {
            int tile_limit = min(BLOCK_SIZE, N - t * BLOCK_SIZE);
            
            #pragma unroll 16
            for (int k = 0; k < tile_limit; ++k) {
                int j = t * BLOCK_SIZE + k;
                if (i != j) {
                    float4 other = tile[k];
                    float dx = my_agent.x - other.x;
                    float dy = my_agent.y - other.y;
                    float dist_sq = dx * dx + dy * dy;

                    // Strict inequality: sqrt(dist_sq) < 5.0 <=> dist_sq < 25.0
                    if (dist_sq < r_sq) {
                        sum_vx += other.z;
                        sum_vy += other.w;
                        neighbor_count++;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Compute updated velocity and position
    if (i < N) {
        float avg_vx = my_agent.z;
        float avg_vy = my_agent.w;

        if (neighbor_count > 0) {
            float inv_count = 1.0f / static_cast<float>(neighbor_count);
            avg_vx = sum_vx * inv_count;
            avg_vy = sum_vy * inv_count;
        }

        float new_vx = my_agent.z + alpha * (avg_vx - my_agent.z);
        float new_vy = my_agent.w + alpha * (avg_vy - my_agent.w);

        float new_x = my_agent.x + new_vx;
        float new_y = my_agent.y + new_vy;

        agents_next[i] = make_float4(new_x, new_y, new_vx, new_vy);
    }
}

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {
    if (N <= 0) return;

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Cast float* to float4* for coalesced 128-bit memory transactions
    const float4* agents_f4 = reinterpret_cast<const float4*>(agents);
    float4* agents_next_f4 = reinterpret_cast<float4*>(agents_next);

    boids_kernel<<<num_blocks, BLOCK_SIZE>>>(agents_f4, agents_next_f4, N);
}
