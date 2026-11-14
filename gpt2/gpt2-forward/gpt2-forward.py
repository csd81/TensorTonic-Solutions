import math
import torch

def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta

def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def causal_mha(
    x: torch.Tensor,
    w_q: torch.Tensor, b_q: torch.Tensor,
    w_k: torch.Tensor, b_k: torch.Tensor,
    w_v: torch.Tensor, b_v: torch.Tensor,
    w_o: torch.Tensor, b_o: torch.Tensor,
    num_heads: int
) -> torch.Tensor:
    B, S, D = x.shape
    head_dim = D // num_heads

    # Transpose weights (.T) for linear projections
    q = x @ w_q.T + b_q
    k = x @ w_k.T + b_k
    v = x @ w_v.T + b_v

    # Reshape and transpose to (B, num_heads, S, head_dim)
    q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply causal mask
    mask = torch.triu(torch.ones((S, S), device=x.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_output = attn_weights @ v

    # Concatenate heads back to (B, S, D) and project output
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
    return attn_output @ w_o.T + b_o

def decoder_block(x: torch.Tensor, block: dict, num_heads: int, eps: float) -> torch.Tensor:
    # 1. Pre-LayerNorm + Causal Multi-Head Attention + Residual
    norm1 = layer_norm(x, block["gamma1"], block["beta1"], eps)
    attn_out = causal_mha(
        norm1,
        block["W_q"], block["b_q"],
        block["W_k"], block["b_k"],
        block["W_v"], block["b_v"],
        block["W_o"], block["b_o"],
        num_heads
    )
    x = x + attn_out

    # 2. Pre-LayerNorm + Feed-Forward Network (MLP) + Residual
    norm2 = layer_norm(x, block["gamma2"], block["beta2"], eps)
    ffn_hidden = gelu(norm2 @ block["W1"].T + block["b1"])
    ffn_out = ffn_hidden @ block["W2"].T + block["b2"]
    x = x + ffn_out

    return x

def gpt2_forward(
    token_ids: torch.Tensor,
    token_embedding: torch.Tensor,
    position_embedding: torch.Tensor,
    blocks: list,
    final_gamma: torch.Tensor,
    final_beta: torch.Tensor,
    num_heads: int,
    eps: float = 1e-5
) -> torch.Tensor:
    # Match input dtype instead of forcing float64 unless needed
    _, S = token_ids.shape

    # 1. Token + Position Embeddings -> shape (B, S, D)
    tok_emb = token_embedding[token_ids]
    pos_emb = position_embedding[:S]
    x = tok_emb + pos_emb

    # 2. Pass through all decoder blocks sequentially
    for block in blocks:
        x = decoder_block(x, block, num_heads, eps)

    # 3. Final LayerNorm
    x = layer_norm(x, final_gamma, final_beta, eps)

    # 4. Weight-tied output projection -> shape (B, S, V)
    logits = x @ token_embedding.T
    return logits
