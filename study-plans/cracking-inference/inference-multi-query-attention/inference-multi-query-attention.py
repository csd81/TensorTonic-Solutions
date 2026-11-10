import math
import torch


def multi_query_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_query_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """Computes multi-query self-attention: independent query heads share one key

    head and one value head.

    Args:
        hidden_states (Tensor): Input tensor of shape (B, S, D)
        w_q (Tensor): Query projection weights of shape (D, D)
        w_k (Tensor): Key projection weights of shape (D, d)
        w_v (Tensor): Value projection weights of shape (D, d)
        w_o (Tensor): Output projection weights of shape (D, D)
        num_query_heads (int): Number of query heads (h)
        causal (bool): Whether to apply an autoregressive/causal mask

    Returns:
        Tensor: The attention output Y with shape (B, S, D)
    """
    B, S, D = hidden_states.shape
    d = D // num_query_heads

    # 1. Project inputs (Matrices store input features along rows, outputs along columns)
    # Shape: (B, S, D) @ (D, D) -> (B, S, D)
    Q = torch.matmul(hidden_states, w_q)
    # Shape: (B, S, D) @ (D, d) -> (B, S, d)
    K = torch.matmul(hidden_states, w_k)
    V = torch.matmul(hidden_states, w_v)

    # 2. Reshape Q into independent heads
    # Q: (B, S, num_query_heads, d) -> Permute to (B, num_query_heads, S, d)
    Q = Q.view(B, S, num_query_heads, d).transpose(1, 2)

    # 3. Add a head dimension to K and V for broadcasting
    # K, V: (B, 1, S, d) so all query heads broadcast over the same KV matrix
    K = K.unsqueeze(1)
    V = V.unsqueeze(1)

    # 4. Compute scaled dot-product attention scores
    # Q: (B, h, S, d), K^T: (B, 1, d, S) -> scores: (B, h, S, S)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)

    # 5. Apply causal mask if requested
    if causal:
        # Create an upper triangular mask matrix of shape (S, S)
        mask = torch.triu(torch.full((S, S), float("-inf"), device=scores.device), diagonal=1)
        scores = scores + mask

    # 6. Softmax over key positions (the last dimension)
    attention_probs = torch.softmax(scores, dim=-1)

    # 7. Compute contextual head outputs
    # attention_probs: (B, h, S, S), V: (B, 1, S, d) -> O: (B, h, S, d)
    O = torch.matmul(attention_probs, V)

    # 8. Concatenate heads and apply final output projection
    # O: (B, h, S, d) -> Permute to (B, S, h, d) -> Flatten to (B, S, D)
    O = O.transpose(1, 2).contiguous().view(B, S, D)

    # Final Linear Projection: (B, S, D) @ (D, D) -> (B, S, D)
    Y = torch.matmul(O, w_o)

    return Y
