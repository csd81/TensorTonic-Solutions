import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
  # 1. Map 2D grid program IDs to output tile coordinates
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  # 2. Compute row, column, and inner-K tile offsets
  offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  offs_k = tl.arange(0, BLOCK_K)

  # 3. Initialize pointer matrices for the first K-block
  a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

  # 4. Initialize an fp32 register accumulator tile
  accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

  # 5. Iterate along the K dimension in chunks of BLOCK_K
  for k in range(0, tl.cdiv(K, BLOCK_K)):
    k_offs = k * BLOCK_K + offs_k

    # Mask boundaries along M, N, and the moving K dimension
    a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
    b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

    # Load operand tiles with out-of-bounds padded to 0.0
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Accumulate partial products using IEEE precision for fp32 consistency
    accumulator = tl.dot(a, b, accumulator, input_precision="ieee")

    # Advance pointers along K for the next iteration
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk

  # 6. Guard non-power-of-two M and N boundaries and write back to global memory
  c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
  c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
  tl.store(c_ptrs, accumulator, mask=c_mask)


def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
  """Launch matmul_kernel: out = A @ B."""
  M, K = A.shape
  K2, N = B.shape
  BLOCK_M = 64
  BLOCK_N = 64
  BLOCK_K = 32
  grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
  matmul_kernel[grid](
      A,
      B,
      out,
      M,
      N,
      K,
      A.stride(0),
      A.stride(1),
      B.stride(0),
      B.stride(1),
      out.stride(0),
      out.stride(1),
      BLOCK_M=BLOCK_M,
      BLOCK_N=BLOCK_N,
      BLOCK_K=BLOCK_K,
  )