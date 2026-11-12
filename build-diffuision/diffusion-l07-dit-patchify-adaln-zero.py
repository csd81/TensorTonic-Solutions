import torch
import torch.nn.functional as F

def dit_block_adaln_zero(x: torch.Tensor, conditioning: torch.Tensor, params: dict, patch_size: int) -> dict:
    """
    Returns a dict with "tokens" and "output", each a tensor of shape (B, N, D) with the dtype of x.
    """
    B, C, H, W = x.shape
    p = patch_size
    grid_h = H // p
    grid_w = W // p
    N = grid_h * grid_w

    # 1. Patchify: (B, C, grid_h, p, grid_w, p) -> (B, grid_h, grid_w, C, p, p) -> (B, N, C * p * p)
    patches = x.view(B, C, grid_h, p, grid_w, p)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(B, N, C * p * p)

    # 2. Linear patch embedding + positional embedding
    tokens = patches @ params["embed_weight"] + params["embed_bias"] + params["pos_embed"]

    # 3. Compute adaptive modulation parameters [shift, scale, gate] from conditioning
    ada_params = F.silu(conditioning) @ params["ada_weight"] + params["ada_bias"]
    shift, scale, gate = torch.chunk(ada_params, chunks=3, dim=-1)

    # Unsqueeze to (B, 1, D) so each parameter broadcasts across the N tokens
    shift = shift.unsqueeze(1)
    scale = scale.unsqueeze(1)
    gate = gate.unsqueeze(1)

    # 4. Parameter-free LayerNorm over feature dimension D (population variance, eps=1e-6)
    D = tokens.shape[-1]
    normed = F.layer_norm(tokens, normalized_shape=(D,), eps=1e-6)

    # 5. Modulate, project through MLP, gate, and add residual connection
    modulated = normed * (1.0 + scale) + shift
    projected = modulated @ params["mlp_weight"] + params["mlp_bias"]
    output = tokens + gate * projected

    return {
        "tokens": tokens,
        "output": output,
    }
