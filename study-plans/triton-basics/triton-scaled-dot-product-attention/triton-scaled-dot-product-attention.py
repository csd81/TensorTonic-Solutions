import torch
import triton
import triton.language as tl


@triton.jit
def matmul_qk_kernel(
    Q_ptr, K_ptr, S_ptr,
    N_Q, N, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    row_mask = offs_m < N_Q
    col_mask = offs_n < N
    d_mask = offs_d < D

    q = tl.load(Q_ptr + offs_m[:, None] * D + offs_d[None, :],
                mask=row_mask[:, None] & d_mask[None, :], other=0.0)
    k = tl.load(K_ptr + offs_n[:, None] * D + offs_d[None, :],
                mask=col_mask[:, None] & d_mask[None, :], other=0.0)
    s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
    tl.store(S_ptr + offs_m[:, None] * N + offs_n[None, :], s,
             mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def softmax_rows_kernel(
    S_ptr, N, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(S_ptr + row * N + cols, mask=mask, other=-1.0e30)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    tl.store(S_ptr + row * N + cols, e / s, mask=mask)


@triton.jit
def matmul_av_kernel(
    S_ptr, V_ptr, out_ptr,
    N_Q, N, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    row_mask = offs_m < N_Q
    d_mask = offs_d < D

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        col_mask = offs_n < N
        s = tl.load(S_ptr + offs_m[:, None] * N + offs_n[None, :],
                    mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        v = tl.load(V_ptr + offs_n[:, None] * D + offs_d[None, :],
                    mask=col_mask[:, None] & d_mask[None, :], other=0.0)
        acc += tl.sum(s[:, :, None] * v[None, :, :], axis=1)

    tl.store(out_ptr + offs_m[:, None] * D + offs_d[None, :], acc,
             mask=row_mask[:, None] & d_mask[None, :])


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, out: torch.Tensor) -> None:
    N_Q, D = Q.shape
    N = K.shape[0]
    scale = 1.0 / (D ** 0.5)
    S = torch.empty((N_Q, N), device=Q.device, dtype=torch.float32)

    BLOCK_M = 1
    BLOCK_N = 16
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2
    BLOCK_N_PADDED = 1
    while BLOCK_N_PADDED < max(N, 1):
        BLOCK_N_PADDED *= 2

    grid_qk = ((N_Q + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    matmul_qk_kernel[grid_qk](
        Q, K, S, N_Q, N, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    softmax_rows_kernel[(N_Q,)](S, N, BLOCK_N=BLOCK_N_PADDED)

    grid_av = ((N_Q + BLOCK_M - 1) // BLOCK_M,)
    matmul_av_kernel[grid_av](
        S, V, out, N_Q, N, D,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
    )