import torch
import math

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as hidden_states (B, S, D).
    """
    B, S, D = hidden_states.shape
    d = D // num_heads

    # 1. Project Q, K, V
    # Shapes: (B, S, D) @ (D, D) -> (B, S, D)
    Q = torch.matmul(hidden_states, w_q)
    K = torch.matmul(hidden_states, w_k)
    V = torch.matmul(hidden_states, w_v)

    # 2. Reshape and transpose to separate heads
    # From (B, S, num_heads, d) -> (B, num_heads, S, d)
    Q = Q.view(B, S, num_heads, d).transpose(1, 2)
    K = K.view(B, S, num_heads, d).transpose(1, 2)
    V = V.view(B, S, num_heads, d).transpose(1, 2)

    # 3. Scaled dot-product attention
    # Q @ K^T -> (B, num_heads, S, S)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)

    # 4. Apply causal mask if requested
    if causal:
        # Create upper triangular mask of -inf
        mask = torch.triu(torch.full((S, S), float('-inf'), device=hidden_states.device), diagonal=1)
        scores = scores + mask

    # Softmax over the last dimension (keys)
    attn_weights = torch.softmax(scores, dim=-1)

    # 5. Multiply by V -> (B, num_heads, S, d)
    attn_output = torch.matmul(attn_weights, V)

    # 6. Concatenate heads back and apply output projection w_o
    # Transpose back to (B, S, num_heads, d) and reshape to (B, S, D)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
    output = torch.matmul(attn_output, w_o)

    return output.to(torch.float32)
