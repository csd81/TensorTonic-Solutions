import torch

def sliding_window_mask(n: int, window: int) -> torch.Tensor:
    """
    Returns an additive causal mask of shape (n, n).
    """
    # Create row (query i) and column (key j) index grids
    idx = torch.arange(n)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # Shape: (n, n), representing i - j

    # Causal condition: j <= i (i.e., i - j >= 0)
    visible = diff >= 0

    # Window condition: i - j < window (only applied when window > 0)
    if window > 0:
        visible = visible & (diff < window)

    # Create float tensor filled with -inf and set visible positions to 0.0
    mask = torch.full((n, n), float("-inf"), dtype=torch.float32)
    mask[visible] = 0.0
    
    return mask
