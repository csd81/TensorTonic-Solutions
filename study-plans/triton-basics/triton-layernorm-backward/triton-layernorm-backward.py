import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_bwd_kernel(
    x_ptr, gamma_ptr, dy_ptr,
    dx_ptr, dgamma_ptr, dbeta_ptr,
    stride_x_row, stride_dy_row, stride_dx_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load the row of x, the row of dy, and the shared gamma. Masked lanes
    # contribute 0.0 so they stay neutral in every reduction below.
    x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + row * stride_dy_row + cols, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0)

    # Recompute the forward statistics instead of loading saved buffers.
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = xc * rstd

    # dy_norm = dy * gamma; c1 and c2 are the two row means in the dx formula.
    dy_norm = dy * gamma
    c1 = tl.sum(dy_norm, axis=0) / N
    c2 = tl.sum(dy_norm * x_hat, axis=0) / N

    # dx is per-row: one coalesced store, no atomics needed.
    dx = rstd * (dy_norm - c1 - x_hat * c2)
    tl.store(dx_ptr + row * stride_dx_row + cols, dx, mask=mask)

    # dgamma/dbeta are cross-row sums: each program atomically adds its row's
    # contribution into the length-N buffers (pre-zeroed by solve).
    tl.atomic_add(dgamma_ptr + cols, dy * x_hat, mask=mask)
    tl.atomic_add(dbeta_ptr + cols, dy, mask=mask)


def solve(
    x: torch.Tensor, gamma: torch.Tensor, dy: torch.Tensor,
    dx_out: torch.Tensor, dgamma_out: torch.Tensor, dbeta_out: torch.Tensor,
    eps: float,
) -> None:
    """Launch the LayerNorm backward kernel: one program per row, atomic reductions for dgamma and dbeta."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    dgamma_out.zero_()
    dbeta_out.zero_()
    grid = (M,)
    layernorm_bwd_kernel[grid](
        x, gamma, dy,
        dx_out, dgamma_out, dbeta_out,
        x.stride(0), dy.stride(0), dx_out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )