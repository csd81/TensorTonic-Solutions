import math


def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    center = size // 2
    denom = 2 * (sigma**2)

    kernel = []
    total_sum = 0.0

    # 1. Compute unnormalized Gaussian weights
    for i in range(size):
        row = []
        y = i - center
        for j in range(size):
            x = j - center
            weight = math.exp(-(x**2 + y**2) / denom)
            row.append(weight)
            total_sum += weight
        kernel.append(row)

    # 2. Normalize every element so the entire kernel sums to 1
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total_sum

    return kernel
