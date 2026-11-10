import numpy as np

def cutmix(image_a: list, image_b: list, bbox: list, label_a: list, label_b: list) -> dict:
    """
    Returns a dictionary with image (C, H, W list), label (K floats), and lam (float).
    """
    # Convert inputs to NumPy float64 arrays to ensure precision
    img_a = np.array(image_a, dtype=np.float64)
    img_b = np.array(image_b, dtype=np.float64)
    lbl_a = np.array(label_a, dtype=np.float64)
    lbl_b = np.array(label_b, dtype=np.float64)
    
    # Extract dimensions from shape (C, H, W)
    C, H, W = img_a.shape
    x1, y1, x2, y2 = bbox
    
    # Calculate the area of the bounding box patch
    patch_area = (x2 - x1) * (y2 - y1)
    
    # Compute lambda (the fraction of image_a that is retained)
    lam = 1.0 - (patch_area / (H * W))
    
    # Create a copy of image_a to avoid mutating the original input
    mixed_img = img_a.copy()
    
    # Replace the bounding box region across all channels with image_b's patch
    mixed_img[:, y1:y2, x1:x2] = img_b[:, y1:y2, x1:x2]
    
    # Interpolate the labels using lambda
    mixed_label = lam * lbl_a + (1.0 - lam) * lbl_b
    
    # Round all values to 4 decimal places as required by the instructions
    return {
        "image": np.round(mixed_img, 4).tolist(),
        "label": np.round(mixed_label, 4).tolist(),
        "lam": float(np.round(lam, 4))
    }
