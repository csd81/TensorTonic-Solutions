import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # -inf sentinel: masked lanes lose the row-max comparison and, after the
    # shift, exp(-inf - max) = 0, so they contribute nothing to the sum.
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=float("-inf"))

    # Stable form: m + log(sum(exp(x - m))).
    row_max = tl.max(x, axis=0)
    sum_exp = tl.sum(tl.exp(x - row_max), axis=0)
    lse = row_max + tl.log(sum_exp)

    # One scalar per row; each program owns exactly one output address.
    tl.store(out_ptr + row, lse)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )
