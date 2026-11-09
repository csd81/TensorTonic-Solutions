import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    N,
    D,
    BLOCK_SIZE: tl.constexpr,
):
  row = tl.program_id(0)
  half_d = D // 2

  # Pair indices and boundary mask
  j = tl.arange(0, BLOCK_SIZE)
  mask = j < half_d

  # Offsets for x and out: row stride is D, pair offsets are 2*j (even) and 2*j + 1 (odd)
  row_x_offset = row * D
  x_even_ptrs = x_ptr + row_x_offset + 2 * j
  x_odd_ptrs = x_ptr + row_x_offset + 2 * j + 1

  # Offsets for cos and sin: row stride is D // 2, column offset is j
  row_table_offset = row * half_d
  cos_ptrs = cos_ptr + row_table_offset + j
  sin_ptrs = sin_ptr + row_table_offset + j

  # Load values into registers
  x_even = tl.load(x_even_ptrs, mask=mask)
  x_odd = tl.load(x_odd_ptrs, mask=mask)
  cos_val = tl.load(cos_ptrs, mask=mask)
  sin_val = tl.load(sin_ptrs, mask=mask)

  # Apply 2D rotation
  out_even = x_even * cos_val - x_odd * sin_val
  out_odd = x_even * sin_val + x_odd * cos_val

  # Write back to output
  row_out_offset = row * D
  out_even_ptrs = out_ptr + row_out_offset + 2 * j
  out_odd_ptrs = out_ptr + row_out_offset + 2 * j + 1

  tl.store(out_even_ptrs, out_even, mask=mask)
  tl.store(out_odd_ptrs, out_odd, mask=mask)


def solve(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor
) -> None:
  """Launch the RoPE kernel: rotate pairs of channels with per-row (cos, sin) tables."""
  N, D = x.shape
  BLOCK_SIZE = 1
  while BLOCK_SIZE < D // 2:
    BLOCK_SIZE *= 2
  if BLOCK_SIZE < 1:
    BLOCK_SIZE = 1
  grid = (N,)
  rope_kernel[grid](x, cos, sin, out, N, D, BLOCK_SIZE=BLOCK_SIZE)