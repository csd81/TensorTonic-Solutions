def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    n = len(X)
    d_in = len(W)
    d_out = len(b)
    
    return [
        [
            float(sum(X[i][k] * W[k][j] for k in range(d_in)) + b[j])
            for j in range(d_out)
        ]
        for i in range(n)
    ]
