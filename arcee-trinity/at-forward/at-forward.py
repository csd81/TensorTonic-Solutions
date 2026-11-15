import math
import torch
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms * weight


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, S, D_h), cos/sin: (S, D_h // 2) or (S, D_h)
    half = x.shape[-1] // 2
    if cos.shape[-1] == half:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D_h)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def _swiglu_ffn(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    # Weights stored as (out_features, in_features)
    gate = F.silu(F.linear(x, w_gate))
    up = F.linear(x, w_up)
    return F.linear(gate * up, w_down)


def _attention_sublayer(
    x: torch.Tensor,
    block: dict,
    layer_idx: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    window_size: int,
    eps: float,
) -> torch.Tensor:
    B, S, D = x.shape
    head_dim = D // num_query_heads
    is_local = (layer_idx % 4) != 3  # 3:1 local (RoPE + SWA) to global (NoPE)

    q = F.linear(x, block["W_q"]).view(B, S, num_query_heads, head_dim)
    k = F.linear(x, block["W_k"]).view(B, S, num_kv_heads, head_dim)
    v = F.linear(x, block["W_v"]).view(B, S, num_kv_heads, head_dim)

    # QK-normalization
    q = _rms_norm(q, block["q_gain"], eps=eps).transpose(1, 2)  # (B, H_q, S, D_h)
    k = _rms_norm(k, block["k_gain"], eps=eps).transpose(1, 2)  # (B, H_kv, S, D_h)
    v = v.transpose(1, 2)  # (B, H_kv, S, D_h)

    if is_local:
        q = _apply_rope(q, cos[:S], sin[:S])
        k = _apply_rope(k, cos[:S], sin[:S])

    # Grouped-Query Attention head expansion
    repeats = num_query_heads // num_kv_heads
    if repeats > 1:
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Causal mask (plus sliding window for local layers)
    idx = torch.arange(S, device=x.device)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # query_pos - key_pos
    mask = diff >= 0
    if is_local:
        mask = mask & (diff < window_size)

    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    sdpa_out = torch.matmul(attn_weights, v)  # (B, H_q, S, D_h)
    sdpa_out = sdpa_out.transpose(1, 2).contiguous().view(B, S, D)

    # Gated attention output before W_o
    gate = torch.sigmoid(F.linear(x, block["W_attn_gate"]))
    return F.linear(sdpa_out * gate, block["W_o"])


def _moe_sublayer(
    x: torch.Tensor,
    block: dict,
    top_k: int,
) -> torch.Tensor:
    num_experts = block["router_weight"].shape[0]
    router_bias = block.get(
        "expert_bias",
        torch.zeros(num_experts, dtype=x.dtype, device=x.device),
    )

    # Sigmoid routing with decoupled bias for Top-K selection
    scores = torch.sigmoid(F.linear(x, block["router_weight"]))  # (B, S, E)
    selection_scores = scores + router_bias
    _, topk_idx = torch.topk(selection_scores, k=top_k, dim=-1)  # (B, S, K)

    topk_weights = torch.gather(scores, dim=-1, index=topk_idx)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if "route_scale" in block:
        topk_weights = topk_weights * block["route_scale"]

    # Always-active shared expert
    out = _swiglu_ffn(
        x,
        block["shared_W_gate"],
        block["shared_W_up"],
        block["shared_W_down"],
    )

    # Routed experts
    for e in range(num_experts):
        mask = topk_idx == e  # (B, S, K)
        if not mask.any():
            continue
        expert_weight = (topk_weights * mask.to(x.dtype)).sum(dim=-1, keepdim=True)
        expert_out = _swiglu_ffn(
            x,
            block["W_gate"][e],
            block["W_up"][e],
            block["W_down"][e],
        )
        out = out + expert_weight * expert_out

    return out


def trinity_forward(
    token_ids: torch.Tensor,
    embedding: torch.Tensor,
    blocks: list,
    cos: torch.Tensor,
    sin: torch.Tensor,
    final_gain: torch.Tensor,
    lm_head: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    top_k: int,
    window_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns vocabulary logits of shape (batch, sequence, vocabulary_size).
    """
    token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    embedding = torch.as_tensor(embedding, dtype=torch.float64)
    cos = torch.as_tensor(cos, dtype=torch.float64)
    sin = torch.as_tensor(sin, dtype=torch.float64)
    final_gain = torch.as_tensor(final_gain, dtype=torch.float64)
    lm_head = torch.as_tensor(lm_head, dtype=torch.float64)

    # 1. Token embedding lookup
    h = embedding[token_ids]

    # 2. Transformer blocks with depth-scaled sandwich norm & 3:1 RoPE/NoPE schedule
    num_layers = len(blocks)
    depth_scale = 1.0 / math.sqrt(num_layers) if num_layers > 0 else 1.0

    for layer_idx, block in enumerate(blocks):
        blk = {
            k: torch.as_tensor(v, dtype=torch.float64) if not isinstance(v, (int, float)) else v
            for k, v in block.items()
        }
        if "trinity_block" in globals():
            h = trinity_block(
                h,
                blk,
                cos,
                sin,
                layer_idx=layer_idx,
                num_layers=num_layers,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                top_k=top_k,
                window_size=window_size,
                eps=eps,
            )
        else:
            # Attention sublayer with depth-scaled sandwich norm
            attn_in = _rms_norm(h, blk["attn_pre_gain"], eps=eps)
            attn_out = _attention_sublayer(
                attn_in,
                blk,
                layer_idx,
                cos,
                sin,
                num_query_heads,
                num_kv_heads,
                window_size,
                eps,
            )
            h = h + _rms_norm(attn_out, blk["attn_post_gain"], eps=eps) * depth_scale

            # MoE sublayer with depth-scaled sandwich norm
            moe_in = _rms_norm(h, blk["moe_pre_gain"], eps=eps)
            moe_out = _moe_sublayer(moe_in, blk, top_k)
            h = h + _rms_norm(moe_out, blk["moe_post_gain"], eps=eps) * depth_scale

    # 3. Final RMSNorm and vocabulary projection
    h_norm = _rms_norm(h, final_gain, eps=eps)
    logits = F.linear(h_norm, lm_head)
    return logits.to(torch.float64)
