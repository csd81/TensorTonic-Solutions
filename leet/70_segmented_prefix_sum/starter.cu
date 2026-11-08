#include <cuda_runtime.h>

#define BLOCK_SIZE 512

struct SegVal {
    double val;
    int flag;
};

__device__ __forceinline__ SegVal seg_op(SegVal a, SegVal b) {
    SegVal res;
    res.val = b.flag ? b.val : (a.val + b.val);
    res.flag = a.flag | b.flag;
    return res;
}

// Warp-level inclusive segmented scan using shuffle intrinsics
__device__ __forceinline__ SegVal warp_scan(SegVal x) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        double other_val = __shfl_up_sync(0xffffffff, x.val, offset);
        int other_flag = __shfl_up_sync(0xffffffff, x.flag, offset);
        if (lane >= offset) {
            x = seg_op({other_val, other_flag}, x);
        }
    }
    return x;
}

// Block-level inclusive segmented scan
__device__ SegVal block_scan(SegVal x, double* s_val, int* s_flag) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    x = warp_scan(x);

    if (lane == 31) {
        s_val[warp_id] = x.val;
        s_flag[warp_id] = x.flag;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = BLOCK_SIZE >> 5;
        SegVal w = (lane < num_warps) ? SegVal{s_val[lane], s_flag[lane]} : SegVal{0.0, 0};
        w = warp_scan(w);
        if (lane < num_warps) {
            s_val[lane] = w.val;
            s_flag[lane] = w.flag;
        }
    }
    __syncthreads();

    if (warp_id > 0) {
        x = seg_op({s_val[warp_id - 1], s_flag[warp_id - 1]}, x);
    }
    return x;
}

// Pass 1: Local inclusive scan on raw inputs, writing block aggregates
__global__ void local_scan_raw_kernel(
    const float* __restrict__ values,
    const int* __restrict__ flags,
    double* __restrict__ inc_scan,
    double* __restrict__ block_vals,
    int* __restrict__ block_flags,
    int N
) {
    __shared__ double s_val[BLOCK_SIZE >> 5];
    __shared__ int s_flag[BLOCK_SIZE >> 5];

    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    SegVal x = (gid < N) ? SegVal{(double)values[gid], flags[gid]} : SegVal{0.0, 0};

    SegVal res = block_scan(x, s_val, s_flag);

    if (gid < N) {
        inc_scan[gid] = res.val;
    }
    if (threadIdx.x == BLOCK_SIZE - 1) {
        block_vals[blockIdx.x] = res.val;
        block_flags[blockIdx.x] = res.flag;
    }
}

// Recursive helper kernel: Local inclusive scan on block aggregates
__global__ void local_scan_agg_kernel(
    const double* __restrict__ in_vals,
    const int* __restrict__ in_flags,
    double* __restrict__ inc_vals,
    int* __restrict__ inc_flags,
    double* __restrict__ next_vals,
    int* __restrict__ next_flags,
    int N
) {
    __shared__ double s_val[BLOCK_SIZE >> 5];
    __shared__ int s_flag[BLOCK_SIZE >> 5];

    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    SegVal x = (gid < N) ? SegVal{in_vals[gid], in_flags[gid]} : SegVal{0.0, 0};

    SegVal res = block_scan(x, s_val, s_flag);

    if (gid < N) {
        inc_vals[gid] = res.val;
        inc_flags[gid] = res.flag;
    }
    if (threadIdx.x == BLOCK_SIZE - 1) {
        next_vals[blockIdx.x] = res.val;
        next_flags[blockIdx.x] = res.flag;
    }
}

// Recursive helper kernel: Propagate scanned super-block prefixes down
__global__ void propagate_agg_kernel(
    double* __restrict__ inc_vals,
    int* __restrict__ inc_flags,
    const double* __restrict__ prev_block_vals,
    const int* __restrict__ prev_block_flags,
    int N
) {
    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (gid >= N) return;

    if (blockIdx.x > 0) {
        SegVal prefix = {prev_block_vals[blockIdx.x - 1], prev_block_flags[blockIdx.x - 1]};
        SegVal cur = {inc_vals[gid], inc_flags[gid]};
        SegVal combined = seg_op(prefix, cur);
        inc_vals[gid] = combined.val;
        inc_flags[gid] = combined.flag;
    }
}

// Pass 3: Apply block prefix, convert inclusive to exclusive, write float output
__global__ void finalize_exclusive_kernel(
    const double* __restrict__ inc_scan,
    const int* __restrict__ flags,
    const double* __restrict__ block_vals,
    float* __restrict__ output,
    int N
) {
    __shared__ double s_inc[BLOCK_SIZE + 1];
    __shared__ int s_has_flag[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    int f = (gid < N) ? flags[gid] : 0;
    double val = (gid < N) ? inc_scan[gid] : 0.0;

    // Determine if any flag == 1 has appeared at or before tid within this block
    // Using a quick inclusive OR scan via warp shuffles + shared memory
    int has_flag = f;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int other = __shfl_up_sync(0xffffffff, has_flag, offset);
        if (lane >= offset) has_flag |= other;
    }
    __shared__ int warp_or[BLOCK_SIZE >> 5];
    if (lane == 31) warp_or[warp_id] = has_flag;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = BLOCK_SIZE >> 5;
        int w = (lane < num_warps) ? warp_or[lane] : 0;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            int other = __shfl_up_sync(0xffffffff, w, offset);
            if (lane >= offset) w |= other;
        }
        if (lane < num_warps) warp_or[lane] = w;
    }
    __syncthreads();

    if (warp_id > 0) has_flag |= warp_or[warp_id - 1];

    // Add incoming block prefix only if no segment reset occurred up to this element
    if (blockIdx.x > 0 && !has_flag) {
        val += block_vals[blockIdx.x - 1];
    }

    // Load boundary element from previous block for exclusive shift at tid == 0
    if (tid == 0) {
        s_inc[0] = (blockIdx.x > 0) ? block_vals[blockIdx.x - 1] : 0.0;
    }
    s_inc[tid + 1] = val;
    __syncthreads();

    if (gid < N) {
        output[gid] = (f == 1) ? 0.0f : (float)s_inc[tid];
    }
}

// Recursive host function to perform inclusive segmented scan on block aggregates
void scan_aggregates(double* d_vals, int* d_flags, int count) {
    int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double* d_next_vals = nullptr;
    int* d_next_flags = nullptr;
    cudaMalloc(&d_next_vals, num_blocks * sizeof(double));
    cudaMalloc(&d_next_flags, num_blocks * sizeof(int));

    local_scan_agg_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_vals, d_flags, d_vals, d_flags, d_next_vals, d_next_flags, count
    );

    if (num_blocks > 1) {
        scan_aggregates(d_next_vals, d_next_flags, num_blocks);
        propagate_agg_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_vals, d_flags, d_next_vals, d_next_flags, count
        );
    }

    cudaFree(d_next_vals);
    cudaFree(d_next_flags);
}

// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {
    if (N <= 0) return;

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double* d_inc_scan = nullptr;
    double* d_block_vals = nullptr;
    int* d_block_flags = nullptr;

    cudaMalloc(&d_inc_scan, (size_t)N * sizeof(double));
    cudaMalloc(&d_block_vals, num_blocks * sizeof(double));
    cudaMalloc(&d_block_flags, num_blocks * sizeof(int));

    // Pass 1: Block-level inclusive segmented scan
    local_scan_raw_kernel<<<num_blocks, BLOCK_SIZE>>>(
        values, flags, d_inc_scan, d_block_vals, d_block_flags, N
    );

    // Pass 2: Scan block aggregates across all blocks
    if (num_blocks > 1) {
        scan_aggregates(d_block_vals, d_block_flags, num_blocks);
    }

    // Pass 3: Propagate prefixes and convert inclusive scan to exclusive output
    finalize_exclusive_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_inc_scan, flags, d_block_vals, output, N
    );

    cudaFree(d_inc_scan);
    cudaFree(d_block_vals);
    cudaFree(d_block_flags);
}
