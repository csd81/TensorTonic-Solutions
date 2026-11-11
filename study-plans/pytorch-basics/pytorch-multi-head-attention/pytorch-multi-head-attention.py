import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Register float32 weight parameters directly as (d_model, d_model)
        # Note: Tests load values directly into self.W_q, W_k, W_v, W_o
        self.W_q = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32))
        self.W_k = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32))
        self.W_v = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32))
        self.W_o = nn.Parameter(torch.empty((d_model, d_model), dtype=torch.float32))
        
        # Standard initialization (overwritten by test harness)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_o)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len_q, _ = Q.shape
        _, seq_len_k, _ = K.shape
        
        # 1. Linear projections: (batch, seq_len, d_model) @ (d_model, d_model)
        Q_proj = Q @ self.W_q
        K_proj = K @ self.W_k
        V_proj = V @ self.W_v
        
        # 2. Split into num_heads contiguous slices of size d_k
        # Shape: (batch, seq_len, num_heads, d_k) -> transpose to (batch, num_heads, seq_len, d_k)
        Q_heads = Q_proj.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K_heads = K_proj.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V_heads = V_proj.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention per head
        # scores shape: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        
        # context shape: (batch, num_heads, seq_len_q, d_k)
        context = torch.matmul(attn_weights, V_heads)
        
        # 4. Concatenate heads back together
        # Transpose back to (batch, seq_len_q, num_heads, d_k) and flatten last two dims
        concat = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # 5. Final output projection
        output = concat @ self.W_o
        
        return output.to(torch.float32)
