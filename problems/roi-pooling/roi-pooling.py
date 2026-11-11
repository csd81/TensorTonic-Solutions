import math

def roi_pool(feature_map: list, rois: list, output_size: int) -> list:
    """
    Returns a list of pooled grids.
    """
    map_h = len(feature_map)
    map_w = len(feature_map[0])
    pooled_rois = []

    for x1, y1, x2, y2 in rois:
        roi_h = y2 - y1
        roi_w = x2 - x1
        grid = []

        for i in range(output_size):
            h_start = y1 + (i * roi_h) // output_size
            h_end = y1 + ((i + 1) * roi_h) // output_size
            if h_end <= h_start:
                h_end = h_start + 1
            h_start = min(h_start, map_h - 1)
            h_end = min(h_end, map_h)

            row = []
            for j in range(output_size):
                w_start = x1 + (j * roi_w) // output_size
                w_end = x1 + ((j + 1) * roi_w) // output_size
                if w_end <= w_start:
                    w_end = w_start + 1
                w_start = min(w_start, map_w - 1)
                w_end = min(w_end, map_w)

                max_val = max(
                    feature_map[r][c]
                    for r in range(h_start, h_end)
                    for c in range(w_start, w_end)
                )
                row.append(max_val)
            grid.append(row)

        pooled_rois.append(grid)

    return pooled_rois
