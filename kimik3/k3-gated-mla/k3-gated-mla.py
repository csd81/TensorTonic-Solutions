import math
import torch
import torch.nn.functional as F

def gated_mla(
    hidden_states: torch.Tensor, 
    query_projection: torch.Tensor, 
    latent_down_projection: torch.Tensor, 
    key_up_projection: torch.Tensor, 
    value_up_projection: torch.Tensor, 
    output_gate_projection: torch.Tensor, 
    output_projection: torch.Tensor, 
    num_heads: int, 
    causal: bool = True
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing gated attention output and the latent key-value cache.
    """
    B, S, D = hidden_states.shape
    D_h = D // num_heads

    # 1. Projections
    # Q = X W_q^T: (B, S, D) @ (D, D) -> (B, S, D)
    Q = torch.matmul(hidden_states, query_projection.transpose(-1, -2))
    
    # C = X W_c^T: (B, S, D) @ (D, L) -> (B, S, L)
    latent_cache = torch.matmul(hidden_states, latent_down_projection.transpose(-1, -2))
    L = latent_cache.shape[-1]

    # Reconstruct K and V from latent cache C
    # K = C W_k^T: (B, S, L) @ (L, D) -> (B, S, D)
    K = torch.matmul(latent_cache, key_up_projection.transpose(-1, -2))
    # V = C W_v^T: (B, S, L) @ (L, D) -> (B, S, D)
    V = torch.matmul(latent_cache, value_up_projection.transpose(-1, -2))

    # 2. Split Q, K, V into heads: (B, H, S, D_h)
    Q = Q.view(B, S, num_heads, D_h).transpose(1, 2)
    K = K.view(B, S, num_heads, D_h).transpose(1, 2)
    V = V.view(B, S, num_heads, D_h).transpose(1, 2)

    # 3. Scaled dot-product attention (NoPE - no positional encoding)
    scale = 1.0 / math.sqrt(D_h)
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale  # (B, H, S, S)

    if causal:
        # Causal mask: upper triangle above diagonal set to -inf
        mask = torch.triu(torch.full((S, S), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)

    attn_weights = F.softmax(scores, dim=-1)
    
    # Context per head: (B, H, S, D_h)
    head_contexts = torch.matmul(attn_weights, V)
    
    # Concatenate heads back to (B, S, D)
    head_contexts = head_contexts.transpose(1, 2).contiguous().view(B, S, D)

    # 4. Channel gate and output projection
    # gate = sigmoid(X W_g^T): (B, S, D)
    gate = torch.sigmoid(torch.matmul(hidden_states, output_gate_projection.transpose(-1, -2)))
    
    # O_tilde = gate * head_contexts
    o_tilde = gate * head_contexts
    
    # Y = O_tilde @ W_o^T: (B, S, D)
    output = torch.matmul(o_tilde, output_projection.transpose(-1, -2))

    return {
        "output": output,
        "latent_cache": latent_cache
    }
