def is_global_layer(layer_idx: int, local_ratio: int) -> bool:
    """
    Returns whether the indexed layer uses global attention.
    """
    cycle_length = local_ratio + 1
    return (layer_idx + 1) % cycle_length == 0
