import torch
import math

def vit_attention(x: list, W_qkv: list, b_qkv: list, W_o: list, b_o: list, num_heads: int) -> list:
    """
    Returns a float list of shape (B, N, D), rounded to 4 decimals.
    """
    # Convert inputs to float64 tensors
    x_t = torch.tensor(x, dtype=torch.float64)
    W_qkv_t = torch.tensor(W_qkv, dtype=torch.float64)
    b_qkv_t = torch.tensor(b_qkv, dtype=torch.float64)
    W_o_t = torch.tensor(W_o, dtype=torch.float64)
    b_o_t = torch.tensor(b_o, dtype=torch.float64)
    
    B, N, D = x_t.shape
    h = num_heads
    d_h = D // h
    
    # 1. Project to Q, K, V combined tensor of shape (B, N, 3D)
    qkv = torch.matmul(x_t, W_qkv_t) + b_qkv_t
    
    # 2. Split into separate Q, K, V tensors each of shape (B, N, D)
    Q, K, V = torch.split(qkv, D, dim=-1)
    
    # 3. Reshape and permute into multi-head format: (B, h, N, d_h)
    Q = Q.reshape(B, N, h, d_h).transpose(1, 2)
    K = K.reshape(B, N, h, d_h).transpose(1, 2)
    V = V.reshape(B, N, h, d_h).transpose(1, 2)
    
    # 4. Compute scaled dot-product attention scores: (B, h, N, N)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)
    
    # 5. Apply Softmax over the key dimension
    attn = torch.softmax(scores, dim=-1)
    
    # 6. Weight the value vectors: (B, h, N, d_h)
    out = torch.matmul(attn, V)
    
    # 7. Concatenate heads back together: (B, N, D)
    out = out.transpose(1, 2).reshape(B, N, D)
    
    # 8. Final linear projection
    y = torch.matmul(out, W_o_t) + b_o_t
    
    # Round to 4 decimal places and convert back to a nested list
    return torch.round(y, decimals=4).tolist()
