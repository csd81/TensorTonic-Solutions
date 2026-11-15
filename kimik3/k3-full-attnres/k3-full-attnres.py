import torch


def full_attention_residual(
    embedding: torch.Tensor,
    previous_outputs: torch.Tensor,
    pseudo_query: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing retrieved representations and depth-attention weights.
    """
    dtype = embedding.dtype
    device = embedding.device

    # 1. Assemble all depth sources: shape (sources, batch, sequence, width)
    # Handle cases where previous_outputs is empty (e.g., shape (0, B, S, D) or an empty list/tensor)
    if previous_outputs is not None and previous_outputs.numel() > 0:
        prev = previous_outputs.to(device=device, dtype=dtype)
        sources = torch.cat([embedding.unsqueeze(0), prev], dim=0)
    else:
        sources = embedding.unsqueeze(0)

    # 2. Apply RMSNorm to form the keys (values remain the unnormalized `sources`)
    rms = torch.sqrt(sources.pow(2).mean(dim=-1, keepdim=True) + eps)
    keys = sources / rms

    # 3. Compute attention logits and softmax weights across depth (dim=0)
    query = pseudo_query.to(device=device, dtype=dtype)
    logits = torch.einsum("d,lbsd->lbs", query, keys)
    attention_weights = torch.softmax(logits, dim=0)

    # 4. Aggregate original source values using the depth weights
    retrieved_representation = torch.einsum(
        "lbs,lbsd->bsd", attention_weights, sources
    )

    return {
        "retrieved_representation": retrieved_representation.to(
            dtype=dtype, device=device
        ),
        "attention_weights": attention_weights.to(dtype=dtype, device=device),
    }
