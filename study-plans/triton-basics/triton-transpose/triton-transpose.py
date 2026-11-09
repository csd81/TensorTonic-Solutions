import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    a_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D Program IDs identifying which tile of A this instance handles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets in A's coordinate system
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 2D boundary mask (valid for both load from A and store into out)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Pointers to load BLOCK_M x BLOCK_N tile from A
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    a_tile = tl.load(a_ptrs, mask=mask, other=0.0)

    # Swapped indices for destination out (out has shape (N, M))
    # out[j, i] = A[i, j], so offs_n maps to out's row axis and offs_m to out's col axis
    out_ptrs = out_ptr + (offs_n[None, :] * stride_om + offs_m[:, None] * stride_on)
    tl.store(out_ptrs, a_tile, mask=mask)


def solve(A: torch.Tensor, out: torch.Tensor) -> None:
    """Launch transpose_kernel: out[j, i] = A[i, j]."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](
        A, out,
        M, N,
        A.stride(0), A.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )