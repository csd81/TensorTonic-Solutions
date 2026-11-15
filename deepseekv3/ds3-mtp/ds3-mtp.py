import torch

def _rms_norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * gain

def mtp_module_input(
    previous_hidden: torch.Tensor,
    future_token_ids: torch.Tensor,
    embedding: torch.Tensor,
    projection: torch.Tensor,
    previous_gain: torch.Tensor,
    embedding_gain: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Returns projected inputs for one multi-token prediction module.
    """
    # 1. Look up embeddings for aligned future tokens: (B, S, D)
    future_emb = embedding[future_token_ids.to(torch.long)]

    # 2. Separately RMS-normalize both streams with their own gains
    norm_hidden = _rms_norm(previous_hidden, previous_gain, eps)
    norm_emb = _rms_norm(future_emb, embedding_gain, eps)

    # 3. Concatenate along model width: (B, S, 2D)
    combined = torch.cat([norm_hidden, norm_emb], dim=-1)

    # 4. Project back to model width D: (B, S, 2D) @ (2D, D) -> (B, S, D)
    out = combined @ projection.T
    return out.to(torch.float64)
