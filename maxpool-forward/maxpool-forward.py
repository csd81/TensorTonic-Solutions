def maxpool_forward(X: list, pool_size: int, stride: int) -> list:
    """
    Returns the maximum value from every pooling window.
    """
    H = len(X)
    W = len(X[0])
    
    # Calculate output dimensions using integer floor division
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    out = []
    
    for i in range(H_out):
        row = []
        for j in range(W_out):
            # Starting coordinates for the current window
            row_start = i * stride
            col_start = j * stride
            
            # Find the maximum value within the p x p window
            max_val = float('-inf')
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[row_start + a][col_start + b]
                    if val > max_val:
                        max_val = val
                        
            row.append(max_val)
        out.append(row)
        
    return out
