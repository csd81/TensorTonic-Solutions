import numpy as np

def mixup(image_a: list, image_b: list, label_a: list, label_b: list, lam: float) -> dict:
    """
    Mixes two images and their labels using the supplied weight lam.
    Returns a dictionary with 'image' (C, H, W float list) and 'label' (K floats).
    All values are rounded to four decimals.
    """
    # Convert input lists to NumPy arrays with float64 precision
    img_a_np = np.array(image_a, dtype=np.float64)
    img_b_np = np.array(image_b, dtype=np.float64)
    lbl_a_np = np.array(label_a, dtype=np.float64)
    lbl_b_np = np.array(label_b, dtype=np.float64)
    
    # Apply the MixUp augmentation formula
    mixed_image = lam * img_a_np + (1.0 - lam) * img_b_np
    mixed_label = lam * lbl_a_np + (1.0 - lam) * lbl_b_np
    
    # Round results to 4 decimal places
    mixed_image = np.round(mixed_image, 4)
    mixed_label = np.round(mixed_label, 4)
    
    # Convert back to standard Python lists for the output dictionary
    return {
        "image": mixed_image.tolist(),
        "label": mixed_label.tolist()
    }
