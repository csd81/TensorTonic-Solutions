import torch
import torch.nn.functional as F

def _rms(x, weight, eps=1e-6):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

def _swiglu(x, W_gate, W_up, W_down):
    gate = x @ W_gate.T
    up = x @ W_up.T
    return (gate * torch.sigmoid(gate) * up) @ W_down.T

def _moe(x, weights, top_k):
    scores = torch.sigmoid(x @ weights["router_weight"].T)
    indices = torch.topk(scores + weights["expert_bias"], top_k, dim=-1).indices
    selected = torch.gather(scores, -1, indices)
    mixing = selected / selected.sum(dim=-1, keepdim=True)
    
    output = _swiglu(x, weights["shared_W_gate"], weights["shared_W_up"], weights["shared_W_down"])
    flat_x = x.reshape(-1, x.shape[-1])
    flat_indices = indices.reshape(-1, top_k)
    flat_mixing = mixing.reshape(-1, top_k)
    flat_output = output.reshape(-1, output.shape[-1])
    
    for token in range(flat_x.shape[0]):
        for slot in range(top_k):
            expert = int(flat_indices[token, slot])
            expert_output = _swiglu(
                flat_x[token], 
                weights["W_gate"][expert], 
                weights["W_up"][expert], 
                weights["W_down"][expert]
            )
            flat_output[token] = flat_output[token] + flat_mixing[token, slot] * expert_output
            
    return flat_output.reshape_as(output)

def _rope(tensor, cos, sin):
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)

def trinity_block(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    weights: dict, 
    layer_idx: int, 
    num_layers: int, 
    num_query_heads: int, 
    num_kv_heads: int, 
    top_k: int, 
    window_size: int, 
    eps: float = 1e-6
) -> torch.Tensor:
    batch, sequence, model_width = x.shape
    head_width = model_width // num_query_heads
    
    attention_input = _rms(x, weights["attn_pre_gain"], eps)
    q = (attention_input @ weights["W_q"].T).reshape(batch, sequence, num_query_heads, head_width).transpose(1, 2)
    k = (attention_input @ weights["W_k"].T).reshape(batch, sequence, num_kv_heads, head_width).transpose(1, 2)
    v = (attention_input @ weights["W_v"].T).reshape(batch, sequence, num_kv_heads, head_width).transpose(1, 2)
    
    q = _rms(q, weights["q_gain"], eps)
    k = _rms(k, weights["k_gain"], eps)
    
    local = layer_idx % 4 < 3
    if local:
        q = _rope(q, cos, sin)
        k = _rope(k, cos, sin)
        
    k = k.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    
    scores = (q @ k.transpose(-2, -1)) / (head_width ** 0.5)
    query = torch.arange(sequence, device=x.device)[:, None]
    key = torch.arange(sequence, device=x.device)[None, :]
    masked = key > query
    if local:
        masked = masked | ((query - key) >= window_size)
    scores = scores.masked_fill(masked, float("-inf"))
    
    heads = torch.softmax(scores, dim=-1) @ v
    gate = torch.sigmoid(attention_input @ weights["W_attn_gate"].T).reshape(batch, sequence, num_query_heads, head_width)
    heads = heads.transpose(1, 2) * gate
    attention_output = heads.reshape(batch, sequence, model_width) @ weights["W_o"].T
    
    hidden = x + _rms(attention_output, weights["attn_post_gain"], eps) / (num_layers ** 0.5)
    moe_input = _rms(hidden, weights["moe_pre_gain"], eps)
    moe_output = _moe(moe_input, weights, top_k)
    
    return hidden + _rms(moe_output, weights["moe_post_gain"], eps) / (num_layers ** 0.5)
