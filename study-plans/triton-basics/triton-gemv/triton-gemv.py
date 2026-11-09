import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Offsets along the M dimension for this program block
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Accumulator vector in fp32
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Base column offsets
    offs_n = tl.arange(0, BLOCK_N)

    # Loop over the N dimension in chunks of BLOCK_N
    for start_n in range(0, N, BLOCK_N):
        current_offs_n = start_n + offs_n
        mask_n = current_offs_n < N

        # 2D pointers for tile of matrix A and 1D pointer for chunk of vector x
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + current_offs_n[None, :] * stride_an)
        x_ptrs = x_ptr + current_offs_n

        # Load tiles with explicit boundary masking and 0.0 fill
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        x_chunk = tl.load(x_ptrs, mask=mask_n, other=0.0)

        # Broadcast vector chunk across rows and compute elementwise product
        prod = a_tile * x_chunk[None, :]

        # Reduce along the column axis (axis=1) and accumulate
        acc += tl.sum(prod, axis=1)

    # Store final row sums back to out
    out_ptrs = out_ptr + offs_m
    tl.store(out_ptrs, acc, mask=mask_m)


def solve(A: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gemv_kernel: out = A @ x."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    gemv_kernel[grid](
        A, x, out,
        M, N,
        A.stride(0), A.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )