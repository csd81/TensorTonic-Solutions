import torch


def gated_attention(
    x: torch.Tensor,
    attention_output: torch.Tensor,
    gate_weight: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
  """Returns gated attention outputs of shape (batch, sequence, model_width)."""
  # x shape: (B, S, D)
  # attention_output shape: (B, H, S, D // H)
  # gate_weight shape: (D, D)
  # output_weight shape: (D, D)

  B, H, S, head_dim = attention_output.shape
  D = x.shape[-1]

  # Compute gate g_t = sigma(W_G * x_t) -> shape: (B, S, D)
  # Linear projection: x @ gate_weight.T (or F.linear(x, gate_weight))
  gate = torch.sigmoid(torch.matmul(x, gate_weight.T))

  # Reshape gate to match attention_output head layout: (B, S, H, head_dim) -> (B, H, S, head_dim)
  gate = gate.view(B, S, H, head_dim).transpose(1, 2)

  # Apply elementwise (Hadamard) product per head: o_tilde = o_t,h * g_t,h
  gated_heads = attention_output * gate

  # Concatenate the gated heads in head order: (B, H, S, head_dim) -> (B, S, H * head_dim) = (B, S, D)
  concatenated = gated_heads.transpose(1, 2).contiguous().view(B, S, D)

  # Apply output projection W_O
  out = torch.matmul(concatenated, output_weight.T)

  return out.to(torch.float64)
