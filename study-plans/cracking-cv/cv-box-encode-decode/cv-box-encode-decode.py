import numpy as np

def encode_and_decode_boxes(proposals: list, gt_boxes: list) -> dict:
    """
    Encodes proposal boxes relative to ground-truth boxes and decodes 
    those offsets back into coordinates.
    
    Boxes are in [left, top, right, bottom] format.
    """
    # Convert inputs to float64 NumPy arrays
    p = np.array(proposals, dtype=np.float64)
    g = np.array(gt_boxes, dtype=np.float64)
    
    # Calculate widths, heights, and centers for proposals
    wp = p[:, 2] - p[:, 0]
    hp = p[:, 3] - p[:, 1]
    cxp = p[:, 0] + wp / 2.0
    cyp = p[:, 1] + hp / 2.0
    
    # Calculate widths, heights, and centers for ground-truth boxes
    wg = g[:, 2] - g[:, 0]
    hg = g[:, 3] - g[:, 1]
    cxg = g[:, 0] + wg / 2.0
    cyg = g[:, 1] + hg / 2.0
    
    # Box Encoding (Deltas)
    tx = (cxg - cxp) / wp
    ty = (cyg - cyp) / hp
    tw = np.log(wg / wp)
    th = np.log(hg / hp)
    deltas = np.stack([tx, ty, tw, th], axis=1)
    
    # Box Decoding
    cxd = cxp + tx * wp
    cyd = cyp + ty * hp
    wd = wp * np.exp(tw)
    hd = hp * np.exp(th)
    
    # Convert decoded centers and sizes back to corner coordinates
    left_d = cxd - wd / 2.0
    top_d = cyd - hd / 2.0
    right_d = cxd + wd / 2.0
    bottom_d = cyd + hd / 2.0
    decoded = np.stack([left_d, top_d, right_d, bottom_d], axis=1)
    
    # Round results to 4 decimal places and convert to nested lists
    return {
        "deltas": np.round(deltas, 4).tolist(),
        "decoded": np.round(decoded, 4).tolist()
    }
