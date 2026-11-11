import torch
import triton
import triton.language as tl

@triton.jit
def matmul_relu_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m_size,
    n_size,
    k_size,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Returns nothing; writes the rectified matrix product through out_ptr.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # fp32 accumulator; loads are cast up so fp16 inputs multiply exactly
    # in float32 (bf16 inputs arrive already converted by the launcher).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(k_size, BLOCK_K)):
        k_remaining = k_size - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < m_size) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < n_size),
            other=0.0,
        )
        acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc,
                     input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ReLU on the fp32 result, before the storage cast.
    acc = tl.maximum(acc, 0.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)

def solve(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matrices must have aligned rank-two shapes")
    if out.shape != (a.shape[0], b.shape[1]):
        raise ValueError("out must have shape (M, N)")
    if not a.is_cuda or not b.is_cuda or not out.is_cuda or a.device != b.device or a.device != out.device:
        raise ValueError("all tensors must be on the same CUDA device")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("a and b must have a matching supported dtype")
    if out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("out has an unsupported dtype")
    m_size, k_size = a.shape
    n_size = b.shape[1]
    if m_size == 0 or n_size == 0:
        return
    if k_size == 0:
        out.zero_()
        return
    kernel_a = a.to(torch.float32) if a.dtype == torch.bfloat16 else a
    kernel_b = b.to(torch.float32) if b.dtype == torch.bfloat16 else b
    kernel_out = torch.empty_like(out, dtype=torch.float32) if out.dtype == torch.bfloat16 else out
    block_m, block_n, block_k = 32, 32, 32
    grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    matmul_relu_kernel[grid](
        kernel_a,
        kernel_b,
        kernel_out,
        m_size,
        n_size,
        k_size,
        kernel_a.stride(0),
        kernel_a.stride(1),
        kernel_b.stride(0),
        kernel_b.stride(1),
        kernel_out.stride(0),
        kernel_out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    if kernel_out is not out:
        out.copy_(kernel_out)
