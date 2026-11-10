import torch
import torch.nn.functional as F

def multi_head_latent_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_down: torch.Tensor,
    w_up_k: torch.Tensor,
    w_up_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> tuple:
    """
    Computes the Multi-Head Latent Attention (MLA) forward pass.
    
    Returns a tuple containing:
      - Y: Attention output tensor of shape (B, S, D)
      - C: Latent representation tensor of shape (B, S, L)
    """
    B, S, D = hidden_states.shape
    L = w_down.shape[1]
    d = D // num_heads

    # 1. Project the shared latent state C
    C = torch.matmul(hidden_states, w_down)  # (B, S, L)

    # 2. Reconstruct full Key and Value states from the compressed latent tensor
    K = torch.matmul(C, w_up_k)  # (B, S, D)
    V = torch.matmul(C, w_up_v)  # (B, S, D)

    # 3. Project Queries
    Q = torch.matmul(hidden_states, w_q)  # (B, S, D)

    # 4. Reshape and permute tensors for multi-head attention: (B, S, h, d) -> (B, h, S, d)
    Q = Q.view(B, S, num_heads, d).transpose(1, 2)
    K = K.view(B, S, num_heads, d).transpose(1, 2)
    V = V.view(B, S, num_heads, d).transpose(1, 2)

    # 5. Compute scaled dot-product attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)

    # 6. Apply causal masking if requested
    if causal:
        mask = torch.triu(torch.full((S, S), float('-inf'), device=hidden_states.device), diagonal=1)
        scores = scores + mask

    # 7. Softmax and attention aggregation
    attn_weights = F.softmax(scores, dim=-1)
    H = torch.matmul(attn_weights, V)  # (B, h, S, d)

    # 8. Concatenate heads back and apply output projection matrix
    H = H.transpose(1, 2).contiguous().view(B, S, D)
    Y = torch.matmul(H, w_o)  # (B, S, D)

    return Y, C
