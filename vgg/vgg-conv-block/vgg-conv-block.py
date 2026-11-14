import numpy as np


def vgg_conv_block(x: np.ndarray, kernels: list, biases: list) -> np.ndarray:
    """
    Returns the float64 block activations in NHWC layout.
    """
    out = np.asarray(x, dtype=np.float64)

    for kernel, bias in zip(kernels, biases):
        k = np.asarray(kernel, dtype=np.float64)
        b = np.asarray(bias, dtype=np.float64)

        kh, kw, _, _ = k.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Zero-pad height (axis 1) and width (axis 2) for 'same' padding
        padded = np.pad(
            out,
            pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

        # Extract spatial patches of shape (N, H, W, C_in, kh, kw)
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, window_shape=(kh, kw), axis=(1, 2)
        )

        # Convolve over (kh, kw, C_in), add bias, and apply ReLU
        conv = np.einsum("nhwcij,ijco->nhwo", windows, k, optimize=True) + b
        out = np.maximum(0.0, conv)

    return out.astype(np.float64)
