import numpy as np

def random_crop(image: np.ndarray, crop_size: int, crop_y: int, crop_x: int) -> np.ndarray:
    """
    Returns the float64 crop at the supplied coordinates.
    """
    arr = np.asarray(image, dtype=np.float64)
    cropped = arr[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size, :]
    return np.array(cropped, dtype=np.float64, copy=True)

def random_horizontal_flip(image: np.ndarray, p: float, flip_rand: float) -> np.ndarray:
    """
    Returns a new image, flipped when flip_rand is less than p.
    """
    arr = np.asarray(image, dtype=np.float64)
    if flip_rand < p:
        flipped = arr[:, ::-1, :]
        return np.array(flipped, dtype=np.float64, copy=True)
    return np.array(arr, dtype=np.float64, copy=True)
