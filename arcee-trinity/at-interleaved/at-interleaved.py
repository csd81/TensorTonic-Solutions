import torch

def interleaved_attention_mask(sequence_length: int, layer_idx: int, window_size: int) -> dict:
    """
    Returns the layer mode and additive attention mask in a dictionary.
    """
    # Every 4th layer (indices 3, 7, 11, ...) is a global NoPE layer; others are local RoPE layers
    uses_rope = (layer_idx % 4) != 3

    # Create coordinate grids for queries (i) and keys (j)
    i = torch.arange(sequence_length).unsqueeze(1)
    j = torch.arange(sequence_length).unsqueeze(0)

    # Causal condition: j <= i
    visible = j <= i

    # Local RoPE layers also restrict attention to a sliding window of size W
    if uses_rope:
        visible = visible & (j >= i - window_size + 1)

    # Build the float64 additive mask (0.0 for visible, -inf for masked)
    mask = torch.full((sequence_length, sequence_length), float("-inf"), dtype=torch.float64)
    mask[visible] = 0.0

    return {
        "uses_rope": uses_rope,
        "mask": mask
    }
