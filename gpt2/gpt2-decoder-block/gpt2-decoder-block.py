import math
import torch

def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * gamma + beta

def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def causal_mha(x: torch.Tensor, weights: dict, num_heads: int) -> torch.Tensor:
    B, S, D = x.shape
    head_dim = D // num_heads

    # Linear projections: (B, S, D)
    Q = x @ weights["W_q"].T + weights["b_q"]
    K = x @ weights["W_k"].T + weights["b_k"]
    V = x @ weights["W_v"].T + weights["b_v"]

    # Reshape and transpose to (B, num_heads, S, head_dim)
    Q = Q.view(B, S, num_heads, head_dim).transpose(1, 2)
    K = K.view(B, S, num_heads, head_dim).transpose(1, 2)
    V = V.view(B, S, num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention scores: (B, num_heads, S, S)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply causal mask (mask out future tokens where j > i)
    causal_mask = torch.triu(torch.ones((S, S), device=x.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    context = attn_weights @ V

    # Concatenate heads back to (B, S, D)
    context = context.transpose(1, 2).contiguous().view(B, S, D)

    # Output projection
    return context @ weights["W_o"].T + weights["b_o"]

def ffn(x: torch.Tensor, weights: dict) -> torch.Tensor:
    hidden = gelu_tanh(x @ weights["W1"].T + weights["b1"])
    return hidden @ weights["W2"].T + weights["b2"]

def gpt2_decoder_block(x: torch.Tensor, weights: dict, num_heads: int, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the transformed residual stream with the same shape as x.
    """
    # Ensure input and weights are float64
    x = x.to(torch.float64)
    w = {k: v.to(torch.float64) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float64, device=x.device) 
         for k, v in weights.items()}

    # 1. Pre-LN + Causal Multi-Head Self-Attention + Residual
    norm1 = layer_norm(x, w["gamma1"], w["beta1"], eps)
    x = x + causal_mha(norm1, w, num_heads)

    # 2. Pre-LN + Feed-Forward Network + Residual
    norm2 = layer_norm(x, w["gamma2"], w["beta2"], eps)
    x = x + ffn(norm2, w)

    return x
