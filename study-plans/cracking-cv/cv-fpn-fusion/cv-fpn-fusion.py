def fpn_top_down_fusion(feature_maps: list) -> list:
    """
    Returns the fused feature maps as nested lists in input order.
    """
    if not feature_maps:
        return []
    
    import numpy as np
    
    # 1. Convert nested lists to NumPy arrays for easier manipulation
    levels = [np.array(f, dtype=np.float64) for f in feature_maps]
    L = len(levels)
    
    # 2. Initialize the top-most layer output
    # P_{L-1} = F_{L-1}
    p_current = levels[-1].copy()
    
    # Track outputs from top-down to preserve unrounded states during accumulation
    p_outputs = [None] * L
    p_outputs[-1] = p_current
    
    # 3. Work downwards from level L-2 down to 0
    for l in range(L - 2, -1, -1):
        f_current = levels[l]
        
        # Perform nearest-neighbor upsampling by a factor of 2 along H and W
        # repeat(2, axis=1) expands H, repeat(2, axis=2) expands W
        p_upsampled = p_current.repeat(2, axis=1).repeat(2, axis=2)
        
        # P_l = F_l + Up2(P_{l+1})
        p_current = f_current + p_upsampled
        p_outputs[l] = p_current

    # 4. Round to 4 decimal places and convert back to nested Python lists
    fused_maps = []
    for p_arr in p_outputs:
        rounded_arr = np.round(p_arr, 4)
        fused_maps.append(rounded_arr.tolist())
        
    return fused_maps
