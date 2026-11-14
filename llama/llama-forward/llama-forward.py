import torch


def rms_norm(value: torch.Tensor, gain: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value * gain / torch.sqrt(torch.mean(value.square(), dim=-1, keepdim=True) + eps)


def rotate_positional_embeddings(value: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    even, odd = value[..., 0::2], value[..., 1::2]
    seq_len = value.shape[-2]
    cos = freqs_cos[:seq_len][None, None, :, :]  # adjust dimensions as needed
    # A simpler slice-based RoPE application matching the prompt's intent:
    cos = freqs_cos[:seq_len][None, None, :]
    sin = freqs_sin[:seq_len][None, None, :]
    output = torch.empty_like(value)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output


def grouped_query_attention(
    hidden: torch.Tensor,
    block: dict,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch, sequence, _ = hidden.shape
    head_width = block["query_projection"].shape[0] // num_query_heads
    
    # 1. RMSNorm on input
    normalized = rms_norm(hidden, block["attention_norm_gain"], eps)
    
    # 2. Q, K, V projections
    query = (normalized @ block["query_projection"].T).view(batch, sequence, num_query_heads, head_width).transpose(1, 2)
    key = (normalized @ block["key_projection"].T).view(batch, sequence, num_kv_heads, head_width).transpose(1, 2)
    value = (normalized @ block["value_projection"].T).view(batch, sequence, num_kv_heads, head_width).transpose(1, 2)
    
    # 3. Apply RoPE
    even_q, odd_q = query[..., 0::2], query[..., 1::2]
    even_k, odd_k = key[..., 0::2], key[..., 1::2]
    
    cos = freqs_cos[:sequence][None, None, :, :] if freqs_cos.dim() == 4 else freqs_cos[:sequence][None, None, :]
    sin = freqs_sin[:sequence][None, None, :, :] if freqs_sin.dim() == 4 else freqs_sin[:sequence][None, None, :]
    
    q_rot = torch.empty_like(query)
    q_rot[..., 0::2] = even_q * freqs_cos - odd_q * freqs_sin
    q_rot[..., 1::2] = even_q * freqs_sin + odd_q * freqs_cos
    
    k_rot = torch.empty_like(key)
    k_rot[..., 0::2] = even_k * freqs_cos - odd_k * freqs_sin
    k_rot[..., 1::2] = even_k * freqs_sin + odd_k * freqs_cos
    
    # 4. Repeat KV heads for Grouped-Query Attention
    repeats = num_query_heads // num_kv_heads
    k_rep = k_rot.repeat_interleave(repeats, dim=1)
    v_rep = value.repeat_interleave(repeats, dim=1)
    
    # 5. Attention scores & mask
    scores = (q_rot @ k_rep.transpose(-2, -1)) / (head_width ** 0.5)
    mask = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool, device=hidden.device), diagonal=1)
    attn_weights = torch.softmax(scores.masked_fill(mask[None, None, :, :], float("-inf")), dim=-1)
    
    attended = attn_weights @ v_rep
    attended = attended.transpose(1, 2).reshape(batch, sequence, -1)
    
    return attended @ block["output_projection"].T


def swiglu_ffn(hidden: torch.Tensor, block: dict, eps: float = 1e-6) -> torch.Tensor:
    normed = rms_norm(hidden, block["ffn_norm_gain"], eps)
    gate = torch.nn.functional.silu(normed @ block["gate_projection"].T)
    up = normed @ block["up_projection"].T
    return (gate * up) @ block["down_projection"].T


def transformer_block(
    h: torch.Tensor,
    block: dict,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    # Self-attention + residual
    h = h + grouped_query_attention(h, block, freqs_cos, freqs_sin, num_query_heads, num_kv_heads, eps)
    # SwiGLU FFN + residual
    h = h + swiglu_ffn(h, block, eps)
    return h


def llama_forward(
    token_ids: torch.Tensor,
    weights: dict,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    embedding = weights["embedding"].to(torch.float64)
    freqs_cos = freqs_cos.to(torch.float64)
    freqs_sin = freqs_sin.to(torch.float64)

    h = embedding[token_ids.long()].to(torch.float64)

    for block in weights["blocks"]:
        block_casted = {k: v.to(torch.float64) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float64) for k, v in block.items()}
        h = transformer_block(
            h,
            block_casted,
            freqs_cos,
            freqs_sin,
            num_query_heads,
            num_kv_heads,
            eps,
        )

    h_final = rms_norm(h, weights["final_norm_gain"].to(torch.float64), eps)
    logits = h_final @ weights["output_projection"].to(torch.float64).T
    return logits.to(torch.float64)
