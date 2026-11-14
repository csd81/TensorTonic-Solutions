import torch


def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns the ordered center-context pairs as an int64 tensor.
    """
    n = token_ids.size(0)
    if n <= 1:
        return torch.empty((0, 2), dtype=torch.int64, device=token_ids.device)

    # Create a (n, n) grid of center indices (i) and context indices (j)
    idx = torch.arange(n, device=token_ids.device)
    i_grid, j_grid = torch.meshgrid(idx, idx, indexing="ij")

    # Mask for valid window positions excluding the center itself
    diff = j_grid - i_grid
    mask = (diff.abs() <= window) & (diff != 0)

    # Gather pairs in row-major (increasing i, then increasing j) order
    centers = token_ids[i_grid[mask]]
    contexts = token_ids[j_grid[mask]]

    return torch.stack((centers, contexts), dim=1).to(torch.int64)
