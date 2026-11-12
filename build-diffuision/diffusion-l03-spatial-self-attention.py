import math
import torch
import torch.nn.functional as F

def spatial_self_attention(x: torch.Tensor, params: dict, num_groups: int, num_heads: int) -> torch.Tensor:
    """
    Returns x plus the attention output, shape (B, C, H, W) with the dtype of x.
    """
    B, C, H, W = x.shape
    N = H * W
    d = C // num_heads

    # 1. Apply Group Normalization (eps=1e-5)
    h = F.group_norm(
        x,
        num_groups=num_groups,
        weight=params["norm_weight"],
        bias=params["norm_bias"],
        eps=1e-5,
    )

    # 2. Convert image feature map (B, C, H, W) -> sequence tokens Z of shape (B, N, C)
    # Pixel (i, j) maps to row i*W + j because flatten(2) scans row-major (H then W)
    z = h.flatten(2).transpose(1, 2)  # (B, H*W, C)

    # 3. Linear projections for Q, K, V
    q = F.linear(z, params["q_weight"], params["q_bias"])  # (B, N, C)
    k = F.linear(z, params["k_weight"], params["k_bias"])  # (B, N, C)
    v = F.linear(z, params["v_weight"], params["v_bias"])  # (B, N, C)

    # 4. Split channels into heads: (B, N, num_heads, d) -> (B, num_heads, N, d)
    q = q.view(B, N, num_heads, d).transpose(1, 2)
    k = k.view(B, N, num_heads, d).transpose(1, 2)
    v = v.view(B, N, num_heads, d).transpose(1, 2)

    # 5. Scaled dot-product attention per head
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)  # (B, num_heads, N, N)
    attn = torch.softmax(scores, dim=-1)                          # softmax over key axis
    out_heads = torch.matmul(attn, v)                             # (B, num_heads, N, d)

    # 6. Concatenate heads back along channel axis: (B, N, C)
    concat = out_heads.transpose(1, 2).contiguous().view(B, N, C)

    # 7. Output linear projection and reshape back to image (B, C, H, W)
    proj = F.linear(concat, params["out_weight"], params["out_bias"])  # (B, N, C)
    out_img = proj.transpose(1, 2).view(B, C, H, W)

    # 8. Residual connection with original input x
    return x + out_img
