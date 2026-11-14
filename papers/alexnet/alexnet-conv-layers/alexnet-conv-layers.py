import numpy as np

def alexnet_conv1(
    image: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray,
    stride: int,
    padding: int,
) -> np.ndarray:
    """Returns the float64 NHWC convolution output."""
    # Convert inputs to float64 to guarantee numerical precision and output dtype
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    B, H, W, C_in = image.shape
    K_h, K_w, _, C_out = kernel.shape

    # Apply symmetric zero-padding to height (axis 1) and width (axis 2)
    if padding > 0:
        padded_image = np.pad(
            image,
            pad_width=((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        padded_image = image

    # Calculate output spatial dimensions
    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1

    # Initialize output tensor
    output = np.zeros((B, H_out, W_out, C_out), dtype=np.float64)

    # Slide the kernel across the padded spatial dimensions
    for i in range(H_out):
        h_start = i * stride
        h_end = h_start + K_h
        for j in range(W_out):
            w_start = j * stride
            w_end = w_start + K_w

            # Extract patch of shape (B, K_h, K_w, C_in)
            patch = padded_image[:, h_start:h_end, w_start:w_end, :]

            # Contract over (K_h, K_w, C_in) -> (B, C_out) and add bias
            output[:, i, j, :] = (
                np.tensordot(patch, kernel, axes=([1, 2, 3], [0, 1, 2])) + bias
            )

    return output
