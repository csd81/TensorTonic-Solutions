import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, gamma_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Program row index
    row_idx = tl.program_id(0)

    # 2. Column offsets and boundary mask for rows where N < BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 3. Row base pointers
    row_x_ptr = x_ptr + row_idx * stride_x_row
    row_out_ptr = out_ptr + row_idx * stride_out_row

    # 4. Load input row and gamma weights
    x = tl.load(row_x_ptr + cols, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0)

    # 5. Compute Mean Square (MS = 1/N * sum(x^2)) in float32 for numerical stability
    x_f32 = x.to(tl.float32)
    sum_sq = tl.sum(x_f32 * x_f32, axis=0)
    mean_sq = sum_sq / N

    # 6. Compute Root Mean Square (RMS)
    rms = tl.sqrt(mean_sq + eps)

    # 7. Normalize, scale by gamma, and store
    x_hat = x_f32 / rms
    y = x_hat * gamma

    tl.store(row_out_ptr + cols, y, mask=mask)


def solve(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the RMSNorm forward kernel; one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_fwd_kernel[grid](
        x, gamma, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )