import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    target_ptr,
    loss_out_ptr,
    stride_logits_row,
    B,
    C,
    BLOCK_SIZE: tl.constexpr,
):
  row_idx = tl.program_id(0)

  # Column offsets and tail mask for non-power-of-two C
  cols = tl.arange(0, BLOCK_SIZE)
  mask = cols < C

  # Base pointer for the current row
  row_logits_ptr = logits_ptr + row_idx * stride_logits_row

  # Load the entire row into registers, padding out-of-bounds lanes with -inf
  logits = tl.load(row_logits_ptr + cols, mask=mask, other=-float("inf"))

  # 1. Stable Log-Sum-Exp: extract maximum logit across valid classes
  row_max = tl.max(logits, axis=0)

  # 2. Subtract max and zero-out masked elements before sum
  exp_logits = tl.where(mask, tl.exp(logits - row_max), 0.0)
  sum_exp = tl.sum(exp_logits, axis=0)
  lse = tl.log(sum_exp) + row_max

  # 3. Retrieve target class index and its logit
  target_idx = tl.load(target_ptr + row_idx)
  target_logit = tl.load(row_logits_ptr + target_idx)

  # 4. Per-row cross-entropy loss: L_i = lse_i - logits[i, t_i]
  loss = lse - target_logit

  # 5. Cross-program reduction into the single-element accumulator
  tl.atomic_add(loss_out_ptr, loss)


def solve(
    logits: torch.Tensor, target: torch.Tensor, loss_out: torch.Tensor
) -> None:
  """Launch the cross-entropy kernel: one program per row, atomic accumulate, then divide by B."""
  B, C = logits.shape
  BLOCK_SIZE = triton.next_power_of_2(C)
  loss_out.zero_()
  grid = (B,)
  cross_entropy_kernel[grid](
      logits,
      target,
      loss_out,
      logits.stride(0),
      B,
      C,
      BLOCK_SIZE=BLOCK_SIZE,
  )
  loss_out.div_(B)