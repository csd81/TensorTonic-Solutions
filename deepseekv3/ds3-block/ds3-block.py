import torch

def _rms(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    return x * gain / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)

def _swiglu(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    gate = x @ W_gate.T
    return (gate * torch.sigmoid(gate) * (x @ W_up.T)) @ W_down.T

def _moe(x: torch.Tensor, weights: dict, top_k: int) -> torch.Tensor:
    scores = torch.sigmoid(x @ weights["router_weight"].T)
    indices = torch.topk(scores + weights["expert_bias"], top_k, dim=-1).indices
    selected = torch.gather(scores, -1, indices)
    mixing = selected / selected.sum(dim=-1, keepdim=True)

    output = _swiglu(x, weights["shared_W_gate"], weights["shared_W_up"], weights["shared_W_down"])

    flat_x = x.reshape(-1, x.shape[-1])
    flat_i = indices.reshape(-1, top_k)
    flat_w = mixing.reshape(-1, top_k)
    flat_o = output.reshape(-1, output.shape[-1])

    for token in range(flat_x.shape[0]):
        for slot in range(top_k):
            expert = int(flat_i[token, slot])
            flat_o[token] = flat_o[token] + flat_w[token, slot] * _swiglu(
                flat_x[token],
                weights["W_gate"][expert],
                weights["W_up"][expert],
                weights["W_down"][expert],
            )
    return flat_o.reshape_as(output)

def _hybrid_ffn(x: torch.Tensor, weights: dict, layer_idx: int, num_dense_layers: int, top_k: int) -> torch.Tensor:
    if layer_idx < num_dense_layers:
        return _swiglu(x, weights["dense_W_gate"], weights["dense_W_up"], weights["dense_W_down"])
    return _moe(x, weights, top_k)

def _rope(x: torch.Tensor, cos_freq: torch.Tensor, sin_freq: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    cos = cos_freq[None, None, :, :]
    sin = sin_freq[None, None, :, :]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)

def _mla(x: torch.Tensor, weights: dict, cos_freq: torch.Tensor, sin_freq: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, sequence, _ = x.shape
    d_content = weights["W_uk"].shape[0] // num_heads
    d_value = weights["W_uv"].shape[0] // num_heads
    d_rope = weights["W_qr"].shape[0] // num_heads

    c_kv = x @ weights["W_dkv"].T
    k_content = (c_kv @ weights["W_uk"].T).reshape(batch, sequence, num_heads, d_content).transpose(1, 2)
    value = (c_kv @ weights["W_uv"].T).reshape(batch, sequence, num_heads, d_value).transpose(1, 2)

    c_query = x @ weights["W_dq"].T
    q_content = (c_query @ weights["W_uq"].T).reshape(batch, sequence, num_heads, d_content).transpose(1, 2)
    q_rope = (c_query @ weights["W_qr"].T).reshape(batch, sequence, num_heads, d_rope).transpose(1, 2)
    k_rope = (x @ weights["W_kr"].T).reshape(batch, sequence, 1, d_rope).transpose(1, 2)

    query = torch.cat([q_content, _rope(q_rope, cos_freq, sin_freq)], dim=-1)
    key = torch.cat([k_content, _rope(k_rope, cos_freq, sin_freq).expand(-1, num_heads, -1, -1)], dim=-1)

    scores = query @ key.transpose(-2, -1) / ((d_content + d_rope) ** 0.5)
    mask = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool, device=x.device), diagonal=1)
    context = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1) @ value

    return context.transpose(1, 2).reshape(batch, sequence, num_heads * d_value) @ weights["W_o"].T

def deepseek_block(
    x: torch.Tensor,
    cos_freq: torch.Tensor,
    sin_freq: torch.Tensor,
    weights: dict,
    layer_idx: int,
    num_dense_layers: int,
    num_heads: int,
    top_k: int,
    eps: float = 1e-6
) -> torch.Tensor:
    hidden = x + _mla(_rms(x, weights["attention_gain"], eps), weights, cos_freq, sin_freq, num_heads)
    return (hidden + _hybrid_ffn(_rms(hidden, weights["ffn_gain"], eps), weights, layer_idx, num_dense_layers, top_k)).to(torch.float64)
