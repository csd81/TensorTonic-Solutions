def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """
    if not data or not data[0]:
        return []

    num_rows = len(data)
    num_cols = len(data[0])

    # Precompute min and range (max - min) for each column
    col_mins = []
    col_ranges = []

    for j in range(num_cols):
        col_values = [data[i][j] for i in range(num_rows)]
        min_val = min(col_values)
        max_val = max(col_values)
        col_mins.append(min_val)
        col_ranges.append(max_val - min_val)

    # Build the scaled matrix
    scaled_data = []
    for i in range(num_rows):
        scaled_row = []
        for j in range(num_cols):
            if col_ranges[j] == 0:
                scaled_row.append(0.0)
            else:
                scaled_val = (data[i][j] - col_mins[j]) / col_ranges[j]
                scaled_row.append(float(scaled_val))
        scaled_data.append(scaled_row)

    return scaled_data
