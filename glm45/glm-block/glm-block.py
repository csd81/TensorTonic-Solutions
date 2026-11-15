import math
import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms) * gamma


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    # x: (n_tokens, num_heads, head_dim)
    # cos, sin: (n_tokens, rope_dim // 2)
    if rope_dim == 0:
        return x

    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    # Reshape to (n_tokens, num_heads, rope_dim // 2, 2) or split halves
    # GLM-4 / Llama style RoPE splits the rotated dimension into first and second half
    half = rope_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]

    cos_b = cos.unsqueeze(1)  # (n_tokens, 1, rope_dim // 2)
    sin_b = sin.unsqueeze(1)  # (n_tokens, 1, rope_dim // 2)

    out1 = x1 * cos_b - x2 * sin_b
    out2 = x2 * cos_b + x1 * sin_b

    return torch.cat([out1, out2, x_pass], dim=-1)


def swiglu(
    x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor
) -> torch.Tensor:
    gate = F.silu(x @ W_gate.T)
    up = x @ W_up.T
    return (gate * up) @ W_down.T


def glm_block(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    weights: dict,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    n_routed_experts: int,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    layer_idx: int,
    first_k_dense_replace: int,
    eps_rms: float = 1e-5,
) -> torch.Tensor:
    """
    Returns one float64 transformer-block output with shape (n_tokens, hidden_size).
    """
    n_tokens, hidden_size = x.shape

    # --- 1. Pre-Norm & Self-Attention (QK-Norm + Partial RoPE + Causal GQA) ---
    h_norm = rms_norm(x, weights["norm1_g"], eps_rms)

    q_dim = num_attention_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    qkv = h_norm @ weights["W_qkv"].T + weights["b_qkv"]
    q, k, v = torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=-1)

    q = q.view(n_tokens, num_attention_heads, head_dim)
    k = k.view(n_tokens, num_kv_heads, head_dim)
    v = v.view(n_tokens, num_kv_heads, head_dim)

    # QK-Norm (RMSNorm over head_dim before RoPE)
    q = rms_norm(q, weights["qk_gamma_q"], eps_rms)
    k = rms_norm(k, weights["qk_gamma_k"], eps_rms)

    # Partial RoPE
    q = apply_partial_rope(q, cos, sin, rope_dim)
    k = apply_partial_rope(k, cos, sin, rope_dim)

    # Grouped-Query Attention (repeat K, V heads to match Q heads)
    n_rep = num_attention_heads // num_kv_heads
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    # Transpose to (num_heads, n_tokens, head_dim) for batched matmul
    q_t = q.transpose(0, 1)
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)

    scores = (q_t @ k_t.transpose(-2, -1)) / math.sqrt(head_dim)

    # Causal mask
    causal_mask = torch.triu(
        torch.full((n_tokens, n_tokens), float("-inf"), dtype=x.dtype, device=x.device),
        diagonal=1,
    )
    scores = scores + causal_mask.unsqueeze(0)

    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = (attn_weights @ v_t).transpose(0, 1).contiguous().view(n_tokens, q_dim)
    attn_out = attn_out @ weights["W_o"].T

    # First Residual Connection
    h = x + attn_out

    # --- 2. Pre-Norm & Feed-Forward Network (Dense SwiGLU vs. Group-Routed MoE) ---
    ffn_in = rms_norm(h, weights["norm2_g"], eps_rms)

    if layer_idx < first_k_dense_replace:
        ffn_out = swiglu(
            ffn_in, weights["W_gate"], weights["W_up"], weights["W_down"]
        )
    else:
        # Always-on Shared Expert
        shared_out = swiglu(
            ffn_in,
            weights["shared_W_gate"],
            weights["shared_W_up"],
            weights["shared_W_down"],
        )

        # Group-Routed MoE
        logits = ffn_in @ weights["gate_W"].T  # (n_tokens, n_routed_experts)
        scores_router = torch.sigmoid(logits)
        if "e_score_correction_bias" in weights:
            scores_for_choice = scores_router + weights["e_score_correction_bias"]
        else:
            scores_for_choice = scores_router

        experts_per_group = n_routed_experts // n_group
        group_scores = scores_for_choice.view(n_tokens, n_group, experts_per_group)

        # Group score: sum of top-2 scores in each group (or top-1 if experts_per_group == 1)
        top_k_in_group = min(2, experts_per_group)
        group_vals = group_scores.topk(top_k_in_group, dim=-1).values.sum(dim=-1)

        # Select topk_group groups
        top_groups = group_vals.topk(topk_group, dim=-1).indices
        group_mask = torch.zeros_like(group_vals, dtype=torch.bool)
        group_mask.scatter_(1, top_groups, True)

        # Mask out experts in unselected groups
        expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(n_tokens, n_routed_experts)
        masked_scores = scores_for_choice.masked_fill(~expert_mask, float("-inf"))

        # Select top num_experts_per_tok experts globally among valid groups
        topk_idx = masked_scores.topk(num_experts_per_tok, dim=-1).indices
        topk_weights = scores_router.gather(1, topk_idx)

        if norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        topk_weights = topk_weights * routed_scaling_factor

        # Compute routed expert outputs
        routed_out = torch.zeros_like(ffn_in)
        for i in range(n_routed_experts):
            tok_mask, k_pos = torch.where(topk_idx == i)
            if tok_mask.numel() > 0:
                expert_in = ffn_in[tok_mask]
                expert_y = swiglu(
                    expert_in,
                    weights["routed_W_gate"][i],
                    weights["routed_W_up"][i],
                    weights["routed_W_down"][i],
                )
                routed_out[tok_mask] += topk_weights[tok_mask, k_pos].unsqueeze(-1) * expert_y

        ffn_out = shared_out + routed_out

    # Second Residual Connection
    y = h + ffn_out
    return y.to(torch.float64)
