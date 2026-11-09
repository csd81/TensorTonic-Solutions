#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

// Block size configuration
#define BLOCK_SIZE 256

__global__ void ppo_loss_kernel(
    const float* __restrict__ advantages, 
    const float* __restrict__ log_pi, 
    const float* __restrict__ log_pi_old,
    float* __restrict__ output, 
    float clip_eps, 
    int total_elements
) {
    // 1. Thread-local accumulation
    float local_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        float adv = advantages[i];
        float l_pi = log_pi[i];
        float l_pi_old = log_pi_old[i];

        // Calculate probability ratio r_bs
        float r = __expf(l_pi - l_pi_old);

        // Clamp the ratio within the trust region
        float r_clamped = fminf(fmaxf(r, 1.0f - clip_eps), 1.0f + clip_eps);

        // Compute the two surrogate objectives
        float surf1 = r * adv;
        float surf2 = r_clamped * adv;

        // PPO chooses the pessimistic (minimum) value
        float loss_token = fminf(surf1, surf2);

        // Accumulate local sum
        local_sum += loss_token;
    }

    // 2. Intra-warp reduction using shuffle intrinsics
    unsigned int mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // 3. Intra-block reduction using shared memory
    __shared__ float shared_sum[BLOCK_SIZE / 32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_sum[warp_id] = local_sum;
    }

    __syncthreads();

    // Final reduction of warp sums by the first warp
    if (warp_id == 0) {
        float block_sum = (threadIdx.x < (blockDim.x / warpSize)) ? shared_sum[lane] : 0.0f;
        unsigned int active_mask = __activemask();
        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(active_mask, block_sum, offset);
        }

        // 4. Atomic consolidation to global output
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Final scale kernel to negate and average the accumulated sum
__global__ void scale_output_kernel(float* output, int total_elements) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = -output[0] / static_cast<float>(total_elements);
    }
}

extern "C" void solve(
    const float* advantages, 
    const float* log_pi, 
    const float* log_pi_old,
    float* output, 
    float clip_eps, 
    int B, 
    int S
) {
    int total_elements = B * S;
    if (total_elements <= 0) return;

    // Initialize output tracking container to 0
    cudaMemset(output, 0, sizeof(float));

    // Dynamic grid sizing balancing hardware occupancy and element count
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024; // Cap grid size to avoid scheduling overhead

    // Launch surrogate reduction kernel
    ppo_loss_kernel<<<num_blocks, BLOCK_SIZE>>>(
        advantages, log_pi, log_pi_old, output, clip_eps, total_elements
    );

    // Launch averaging and negation pass
    scale_output_kernel<<<1, 1>>>(output, total_elements);
}
