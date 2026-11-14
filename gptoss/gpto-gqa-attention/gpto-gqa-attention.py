import torch

def gpto_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
    sliding_window: int = 0
) -> torch.Tensor:
    """
    Returns flattened grouped-query attention outputs for all tokens.
    """
    n, H_kv, g, d = Q.shape
    device = Q.device
    dtype = Q.dtype

    # 1. Compute raw scaled attention scores
    # Q: (n, H_kv, g, d), K: (n, H_kv, d) -> scores: (H_kv, g, n, n)
    scores = torch.einsum("ihmd,jhd->hmij", Q, K) * sm_scale

    # 2. Build causal and optional sliding-window mask
    q_idx = torch.arange(n, device=device)[:, None]  # (n, 1)
    k_idx = torch.arange(n, device=device)[None, :]  # (1, n)

    # Causal mask: query i can only attend to keys j <= i
    valid = k_idx <= q_idx

    # Sliding window mask: if sliding_window > 0, keep at most `sliding_window` tokens
    if sliding_window > 0:
        valid = valid & (q_idx - k_idx < sliding_window)

    # Apply -inf to masked positions
    scores = scores.masked_fill(~valid[None, None, :, :], float("-inf"))

    # 3. Reshape sinks to match head dimensions (H_kv, g, n, 1) and concatenate
    sink_logits = sinks.view(H_kv, g, 1, 1).expand(H_kv, g, n, 1)
    logits_with_sink = torch.cat([scores, sink_logits], dim=-1)  # (H_kv, g, n, n + 1)

    # 4. Numerically stable softmax over tokens + sink, then drop the sink weight
    probs = torch.softmax(logits_with_sink, dim=-1)[..., :-1]  # (H_kv, g, n, n)

    # 5. Weighted sum of values: probs (H_kv, g, n, n), V (n, H_kv, d) -> out (n, H_kv, g, d)
    out = torch.einsum("hmij,jhd->ihmd", probs, V)

    # 6. Flatten head dimensions to (n, H_kv * g * d)
    return out.reshape(n, H_kv * g * d).to(torch.float64)
