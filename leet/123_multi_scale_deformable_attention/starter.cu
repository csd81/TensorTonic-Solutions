#include <cuda_runtime.h>

__global__ void ms_deformable_attn_kernel(
    const float* __restrict__ value,
    const int* __restrict__ spatial_shapes,
    const float* __restrict__ sampling_loc,
    const float* __restrict__ attn_weight,
    float* __restrict__ output,
    int num_queries, int num_heads, int head_dim, int num_levels, int num_points) 
{
    // Threads map to (query, head, dim)
    int q = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;

    if (q >= num_queries || h >= num_heads || d >= head_dim) return;

    // Dynamically calculate level_start offsets for up to 8 levels
    int level_start[8];
    int current_offset = 0;
    for (int l = 0; l < num_levels; ++l) {
        level_start[l] = current_offset;
        current_offset += spatial_shapes[l * 2] * spatial_shapes[l * 2 + 1];
    }

    float sum = 0.0f;

    // Linear indexing stride variables for sampling_loc and attn_weight
    // sampling_loc shape: (num_queries, num_heads, num_levels, num_points, 2)
    // attn_weight shape: (num_queries, num_heads, num_levels, num_points)
    int base_sample_idx = (((q * num_heads + h) * num_levels) * num_points) * 2;
    int base_weight_idx = ((q * num_heads + h) * num_levels) * num_points;

    for (int l = 0; l < num_levels; ++l) {
        int H_l = spatial_shapes[l * 2];
        int W_l = spatial_shapes[l * 2 + 1];
        int lvl_start = level_start[l];

        for (int p = 0; p < num_points; ++p) {
            int sample_offset = base_sample_idx + (l * num_points + p) * 2;
            float loc_x = sampling_loc[sample_offset];
            float loc_y = sampling_loc[sample_offset + 1];

            // Map normalized location to the pixel grid with half-pixel convention
            float x = loc_x * W_l - 0.5f;
            float y = loc_y * H_l - 0.5f;

            int x0 = __float2int_rd(x);
            int y0 = __float2int_rd(y);

            float dx = x - x0;
            float dy = y - y0;

            float w00 = (1.0f - dy) * (1.0f - dx);
            float w01 = (1.0f - dy) * dx;
            float w10 = dy * (1.0f - dx);
            float w11 = dy * dx;

            float interpolated_val = 0.0f;

            // Bilinear interpolation checking bounds per pixel corner
            // Top-Left (y0, x0)
            if (y0 >= 0 && y0 < H_l && x0 >= 0 && x0 < W_l) {
                int idx = (lvl_start + y0 * W_l + x0) * num_heads * head_dim + h * head_dim + d;
                interpolated_val += w00 * value[idx];
            }
            // Top-Right (y0, x0 + 1)
            if (y0 >= 0 && y0 < H_l && (x0 + 1) >= 0 && (x0 + 1) < W_l) {
                int idx = (lvl_start + y0 * W_l + (x0 + 1)) * num_heads * head_dim + h * head_dim + d;
                interpolated_val += w01 * value[idx];
            }
            // Bottom-Left (y0 + 1, x0)
            if ((y0 + 1) >= 0 && (y0 + 1) < H_l && x0 >= 0 && x0 < W_l) {
                int idx = (lvl_start + (y0 + 1) * W_l + x0) * num_heads * head_dim + h * head_dim + d;
                interpolated_val += w10 * value[idx];
            }
            // Bottom-Right (y0 + 1, x0 + 1)
            if ((y0 + 1) >= 0 && (y0 + 1) < H_l && (x0 + 1) >= 0 && (x0 + 1) < W_l) {
                int idx = (lvl_start + (y0 + 1) * W_l + (x0 + 1)) * num_heads * head_dim + h * head_dim + d;
                interpolated_val += w11 * value[idx];
            }

            int weight_idx = base_weight_idx + l * num_points + p;
            sum += attn_weight[weight_idx] * interpolated_val;
        }
    }

    // Write final accumulated feature row into output
    int out_idx = (q * num_heads + h) * head_dim + d;
    output[out_idx] = sum;
}

extern "C" void solve(const float* value, const int* spatial_shapes, const float* sampling_loc,
                      const float* attn_weight, float* output, int num_queries, int num_heads,
                      int head_dim, int num_levels, int num_points) 
{
    // Define thread block matching the head_dim dimension
    dim3 threadsPerBlock(head_dim, 1, 1);
    
    // Grid maps along number of heads (x-axis) and queries (y-axis)
    dim3 numBlocks(num_heads, num_queries, 1);

    ms_deformable_attn_kernel<<<numBlocks, threadsPerBlock>>>(
        value, spatial_shapes, sampling_loc, attn_weight, output,
        num_queries, num_heads, head_dim, num_levels, num_points
    );
}
