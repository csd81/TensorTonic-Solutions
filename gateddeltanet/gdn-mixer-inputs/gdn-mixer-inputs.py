import torch
import torch.nn.functional as F
from typing import Optional

def gdn_mixer_inputs(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    conv_q: torch.Tensor,
    conv_k: torch.Tensor,
    conv_v: torch.Tensor,
    w_alpha: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    w_beta: torch.Tensor,
    b_beta: torch.Tensor,
    num_heads: int,
    conv_cache: Optional[tuple] = None,
    eps: float = 1e-6
) -> tuple:
    """
    Returns queries, keys, values, log-decays, write strengths and the new convolution cache.
    """
    B, L, _ = hidden_states.shape
    Kc = conv_q.shape[1]

    def process_branch(w_proj: torch.Tensor, conv_w: torch.Tensor, cache: Optional[torch.Tensor]):
        # 1. Linear projection: (B, L, d) @ (d, C) -> (B, L, C)
        p = hidden_states @ w_proj
        C = p.shape[-1]

        # 2. Prepend cache or zeros of length Kc - 1
        if cache is None:
            prefix = torch.zeros((B, Kc - 1, C), dtype=p.dtype, device=p.device)
        else:
            prefix = cache

        p_ext = torch.cat([prefix, p], dim=1)  # (B, Kc - 1 + L, C)

        # 3. Extract new cache: last Kc - 1 rows before convolution
        new_cache = p_ext[:, -(Kc - 1):, :].clone()

        # 4. Depthwise causal cross-correlation via conv1d
        # conv1d expects (B, C, T) and weights of shape (C, 1, Kc)
        y = F.conv1d(
            p_ext.transpose(1, 2),
            conv_w.unsqueeze(1),
            groups=C
        ).transpose(1, 2)  # (B, L, C)

        # 5. SiLU activation
        return F.silu(y), new_cache

    q_cache = conv_cache[0] if conv_cache is not None else None
    k_cache = conv_cache[1] if conv_cache is not None else None
    v_cache = conv_cache[2] if conv_cache is not None else None

    q, new_q_cache = process_branch(w_q, conv_q, q_cache)
    k, new_k_cache = process_branch(w_k, conv_k, k_cache)
    v, new_v_cache = process_branch(w_v, conv_v, v_cache)

    # Reshape into (B, L, H, d_head)
    d_k = q.shape[-1] // num_heads
    d_v = v.shape[-1] // num_heads

    q = q.view(B, L, num_heads, d_k)
    k = k.view(B, L, num_heads, d_k)
    v = v.view(B, L, num_heads, d_v)

    # Per-head L2 normalization for queries and keys
    q = q / torch.sqrt(torch.sum(q ** 2, dim=-1, keepdim=True) + eps)
    k = k / torch.sqrt(torch.sum(k ** 2, dim=-1, keepdim=True) + eps)

    # Log-decay and write strength (linear projections only)
    alpha_logits = hidden_states @ w_alpha + dt_bias
    g = -torch.exp(a_log) * F.softplus(alpha_logits)

    beta_logits = hidden_states @ w_beta + b_beta
    beta = torch.sigmoid(beta_logits)

    return q, k, v, g, beta, (new_q_cache, new_k_cache, new_v_cache)
