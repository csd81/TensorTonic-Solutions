import math


def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
    """
    Returns a two-dimensional list with shape (new_h, new_w).
    """
    h = len(image)
    w = len(image[0])

    output = [[0.0] * new_w for _ in range(new_h)]

    for i in range(new_h):
        # Map output y-coordinate to source grid (0 if new_h == 1)
        y = 0.0 if new_h == 1 else i * (h - 1) / (new_h - 1)
        y0 = math.floor(y)
        y1 = min(y0 + 1, h - 1)
        dy = y - y0

        for j in range(new_w):
            # Map output x-coordinate to source grid (0 if new_w == 1)
            x = 0.0 if new_w == 1 else j * (w - 1) / (new_w - 1)
            x0 = math.floor(x)
            x1 = min(x0 + 1, w - 1)
            dx = x - x0

            # Horizontal linear interpolations
            v0 = image[y0][x0] * (1.0 - dx) + image[y0][x1] * dx
            v1 = image[y1][x0] * (1.0 - dx) + image[y1][x1] * dx

            # Vertical linear interpolation
            output[i][j] = v0 * (1.0 - dy) + v1 * dy

    return output
