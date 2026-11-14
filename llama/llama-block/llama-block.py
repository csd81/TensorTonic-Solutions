import math
import torch
import torch.nn.functional as F

def rms_norm(x: torch.Tensor, gain: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * gain

def apply_rope(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    S = x.shape[2]
    head_dim = x.shape[-1]
    cos = freqs_cos[:S].view(1, 1, S, head_dim // 2)
    sin = freqs_sin[:S].view(1, 1, S, head_dim // 2)
    
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return out

def grouped_query_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    B, S, D = x.shape
    head_dim = D // num_query_heads
    
    q = x @ w_q.T
    k = x @ w_k.T
    v = x @ w_v.T
    
    q = q.view(B, S, num_query_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    
    q = apply_rope(q, freqs_cos, freqs_sin)
    k = apply_rope(k, freqs_cos, freqs_sin)
    
    n_rep = num_query_heads // num_kv_heads
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    causal_mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn_probs = torch.softmax(scores, dim=-1)
    context = attn_probs @ v
    context = context.transpose(1, 2).contiguous().view(B, S, D)
    return context @ w_o.T

def swiglu_ffn(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    gate = F.silu(x @ w_gate.T)
    up = x @ w_up.T
    return (gate * up) @ w_down.T

def llama_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    x = x.to(torch.float64)
    
    # Map incoming dictionary keys to the expected internal names
    w_q = weights.get("wq", weights.get("query_projection"))
    w_k = weights.get("wk", weights.get("key_projection"))
    w_v = weights.get("wv", weights.get("value_projection"))
    w_o = weights.get("wo", weights.get("output_projection"))
    w_gate = weights.get("w_gate", weights.get("gate_projection"))
    w_up = weights.get("w_up", weights.get("up_projection"))
    w_down = weights.get("w_down", weights.get("down_projection"))
    
    # 1. Pre-norm Attention + Residual
    x_norm = rms_norm(x, weights["attention_norm_gain"], eps=eps)
    attn_out = grouped_query_attention(
        x=x_norm,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
    )
    h = x + attn_out
    
    # 2. Pre-norm SwiGLU FFN + Residual
    h_norm = rms_norm(h, weights["ffn_norm_gain"], eps=eps)
    ffn_out = swiglu_ffn(
        x=h_norm,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
    )
    y = h + ffn_out
    return y.to(torch.float64)
