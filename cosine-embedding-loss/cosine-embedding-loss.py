import math


def cosine_embedding_loss(
    x1: list, x2: list, label: int, margin: float
) -> float:
    """
    Returns the cosine embedding loss as a float.
    """
    dot_product = sum(a * b for a, b in zip(x1, x2))
    norm_x1 = math.sqrt(sum(a * a for a in x1))
    norm_x2 = math.sqrt(sum(b * b for b in x2))

    cos_sim = dot_product / (norm_x1 * norm_x2)

    if label == 1:
        return float(1.0 - cos_sim)
    else:
        return float(max(0.0, cos_sim - margin))
