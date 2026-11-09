import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Map program ID to row
    row_idx = tl.program_id(0)

    # Pointers to current row
    row_start_x = x_ptr + row_idx * stride_x_row
    row_start_out = out_ptr + row_idx * stride_out_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load input row (masked lanes padded with 0.0)
    x = tl.load(row_start_x + cols, mask=mask, other=0.0)

    # 1. Compute mean: μ = (1 / N) * Σ x
    mean = tl.sum(x, axis=0) / N

    # 2. Center values and zero out masked lanes
    x_centered = x - mean
    x_centered = tl.where(mask, x_centered, 0.0)

    # 3. Compute variance: σ² = (1 / N) * Σ (x - μ)²
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    # 4. Normalize
    x_hat = x_centered * rstd

    # 5. Load affine parameters and apply transform
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0)
    out = x_hat * gamma + beta

    # Store normalized result
    tl.store(row_start_out + cols, out, mask=mask)


def solve(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the LayerNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    layernorm_fwd_kernel[grid](
        x, gamma, beta, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )