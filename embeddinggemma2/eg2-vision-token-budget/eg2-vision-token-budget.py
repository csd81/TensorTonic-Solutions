import math
import torch

def pool_to_budget(patch_features: torch.Tensor, budget: int) -> tuple[torch.Tensor, int]:
    """
    Returns a tuple of the pooled soft tokens and the window size.
    """
    G_h, G_w, D = patch_features.shape

    # 1. Find the smallest positive integer k satisfying ceil(G_h / k) * ceil(G_w / k) <= budget
    k = 1
    while math.ceil(G_h / k) * math.ceil(G_w / k) > budget:
        k += 1

    # Fast path when no spatial pooling is needed
    if k == 1:
        return patch_features.reshape(G_h * G_w, D).clone(), 1

    # 2. Pool each window in row-major order
    num_rows = math.ceil(G_h / k)
    num_cols = math.ceil(G_w / k)
    
    pooled_tokens = torch.empty(
        (num_rows * num_cols, D),
        dtype=patch_features.dtype,
        device=patch_features.device,
    )

    idx = 0
    for a in range(num_rows):
        r_start = a * k
        r_end = min(r_start + k, G_h)
        for b in range(num_cols):
            c_start = b * k
            c_end = min(c_start + k, G_w)
            
            # Slice the window and average over both spatial dimensions (0 and 1)
            window = patch_features[r_start:r_end, c_start:c_end, :]
            pooled_tokens[idx] = window.mean(dim=(0, 1))
            idx += 1

    return pooled_tokens, k
