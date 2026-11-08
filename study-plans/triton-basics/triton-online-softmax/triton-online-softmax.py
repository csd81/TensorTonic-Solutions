import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * N

    # Pass 1: stream the row in chunks, maintaining the running max and the
    # running denominator anchored at that max. -1e30 is the neutral start;
    # padded lanes load -1e30 so they never become the chunk max.
    m = -1e30
    l = 0.0
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=-1e30)

        m_new = tl.maximum(m, tl.max(x, axis=0))
        # Re-anchor the old denominator to m_new, then add this chunk.
        l = l * tl.exp(m - m_new) + tl.sum(
            tl.where(mask, tl.exp(x - m_new), 0.0), axis=0
        )
        m = m_new

    # Pass 2: reload the chunks and write exp(x - m) / l with the final state.
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        tl.store(out_ptr + row_start + cols, tl.exp(x - m) / l, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the online softmax kernel row-wise."""
    M, N = x.shape
    BLOCK_SIZE = 1024
    grid = (M,)
    online_softmax_kernel[grid](x, out, M, N, BLOCK_SIZE=BLOCK_SIZE)
