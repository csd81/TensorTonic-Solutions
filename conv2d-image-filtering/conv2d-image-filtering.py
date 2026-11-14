def conv2d(image: list, kernel: list, stride: int = 1, padding: int = 0) -> list:
    """
    Applies a 2D convolution (cross-correlation) to a single-channel image.
    Returns a two-dimensional list.
    """
    H = len(image)
    W = len(image[0])
    Kh = len(kernel)
    Kw = len(kernel[0])

    # 1. Pad the image with zeros on all four sides
    padded_H = H + 2 * padding
    padded_W = W + 2 * padding
    padded_image = [[0] * padded_W for _ in range(padded_H)]
    
    for r in range(H):
        for c in range(W):
            padded_image[r + padding][c + padding] = image[r][c]

    # 2. Calculate output dimensions
    H_out = (H + 2 * padding - Kh) // stride + 1
    W_out = (W + 2 * padding - Kw) // stride + 1

    # 3. Initialize output matrix and slide the kernel
    output = [[0] * W_out for _ in range(H_out)]

    for i in range(H_out):
        for j in range(W_out):
            total = 0
            # Element-wise multiplication and summation
            for a in range(Kh):
                for b in range(Kw):
                    total += padded_image[i * stride + a][j * stride + b] * kernel[a][b]
            output[i][j] = total

    return output
