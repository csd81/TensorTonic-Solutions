import math
import torch


def _rms_norm(x: torch.Tensor, g: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * rms * g


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (n_tokens, n_heads, head_dim)
    # cos, sin: (n_tokens, head_dim // 2)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


def _gptoss_swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    # Interleaved even (gate) and odd (linear/up) channels
    x_glu = x[..., ::2].clamp(min=None, max=limit)
    x_linear = x[..., 1::2].clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1.0)


def gpto_forward(
    input_ids: torch.Tensor,
    embed_W: torch.Tensor,
    blocks: list[dict],
    final_norm_g: torch.Tensor,
    lm_head_W: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_kv_heads: int,
    q_mult: int,
    head_dim: int,
    top_k: int,
    sliding_window: int,
    swiglu_limit: float = 7.0,
    eps_rms: float = 1e-5,
) -> torch.Tensor:
    """
    Returns vocabulary logits of shape (n_tokens, vocab_size).
    """
    x = embed_W[input_ids].to(torch.float64)
    n_tokens, hidden_size = x.shape
    n_q_heads = n_kv_heads * q_mult
    sm_scale = 1.0 / math.sqrt(head_dim)

    for layer_idx, block in enumerate(blocks):
        # 1. Pre-norm Grouped-Query Attention with RoPE & Learned Sinks
        h = _rms_norm(x, block["attn_norm_g"], eps_rms)

        q = (h @ block["W_q"].T).view(n_tokens, n_q_heads, head_dim)
        k = (h @ block["W_k"].T).view(n_tokens, n_kv_heads, head_dim)
        v = (h @ block["W_v"].T).view(n_tokens, n_kv_heads, head_dim)

        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # Group queries to match KV heads: (n_tokens, n_kv_heads, q_mult, head_dim)
        q = q.view(n_tokens, n_kv_heads, q_mult, head_dim)
        k_exp = k[:, :, None, :].expand(-1, -1, q_mult, -1)
        v_exp = v[:, :, None, :].expand(-1, -1, q_mult, -1)

        # Attention mask: even blocks use sliding window, odd blocks use full causal
        mask = torch.triu(x.new_full((n_tokens, n_tokens), float("-inf")), diagonal=1)
        if layer_idx % 2 == 0 and sliding_window > 0:
            mask = mask + torch.tril(
                x.new_full((n_tokens, n_tokens), float("-inf")), diagonal=-sliding_window
            )

        # Scaled dot-product scores + learned per-head attention sink logit
        qk = torch.einsum("qhmd,khmd->hmqk", q, k_exp) * sm_scale
        qk = qk + mask[None, None, :, :]
        sinks = block["sinks"].reshape(n_kv_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
        qk_with_sink = torch.cat([qk, sinks], dim=-1)

        # Softmax includes the sink, then drop the sink weight before aggregating V
        attn_weights = torch.softmax(qk_with_sink, dim=-1)[..., :-1]
        attn = torch.einsum("hmqk,khmd->qhmd", attn_weights, v_exp).reshape(
            n_tokens, n_q_heads * head_dim
        )

        x = x + (attn @ block["W_o"].T)

        # 2. Pre-norm Sparse Mixture-of-Experts (MoE)
        h = _rms_norm(x, block["mlp_norm_g"], eps_rms)
        router_logits = h @ block["gate_W"].T

        topk_vals, topk_idx = torch.topk(router_logits, k=top_k, dim=-1, sorted=True)
        router_weights = torch.softmax(topk_vals, dim=-1)

        experts = block["experts"]
        moe_out = torch.zeros_like(h)
        for t_idx in range(n_tokens):
            token_h = h[t_idx]
            for k_idx in range(top_k):
                exp = experts[int(topk_idx[t_idx, k_idx].item())]
                gu = token_h @ exp["W_gate_up"].T
                act = _gptoss_swiglu(gu, limit=swiglu_limit)
                down = act @ exp["W_down"].T
                moe_out[t_idx] += router_weights[t_idx, k_idx] * down

        x = x + moe_out

    # 3. Final RMSNorm & LM Head Projection
    x = _rms_norm(x, final_norm_g, eps_rms)
    logits = x @ lm_head_W
    return logits.to(torch.float64)
