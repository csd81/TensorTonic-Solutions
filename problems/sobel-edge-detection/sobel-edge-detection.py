import numpy as np


def sobel_edges(image):
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    padded = np.pad(img, pad_width=1, mode="constant", constant_values=0)

    Gx = np.zeros((H, W), dtype=np.float64)
    Gy = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            region = padded[i : i + 3, j : j + 3]
            Gx[i, j] = np.sum(region * Kx)
            Gy[i, j] = np.sum(region * Ky)

    G = np.sqrt(Gx**2 + Gy**2)
    return G.tolist()
