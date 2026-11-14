import torch

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Returns rotated query and key tensors in a dictionary.
    """
    def rotate_tensor(x: torch.Tensor) -> torch.Tensor:
        # Reshape (B, H, S, D_h) -> (B, H, S, D_h // 2, 2) to isolate consecutive pairs
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        x_even = x_pairs[..., 0]  # x_{2m}
        x_odd = x_pairs[..., 1]   # x_{2m+1}

        # Broadcast freqs from (S, D_h // 2) -> (1, 1, S, D_h // 2)
        cos = freqs_cos.unsqueeze(0).unsqueeze(0)
        sin = freqs_sin.unsqueeze(0).unsqueeze(0)

        # Apply the 2D rotation formulas
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Interleave the pairs back together and flatten to (B, H, S, D_h)
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
        return out.to(torch.float64)

    return {
        "query": rotate_tensor(q),
        "key": rotate_tensor(k),
    }
