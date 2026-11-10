import math

def distance_metric(x: list, y: list, metric: str, p: float = 2) -> float:
    """
    Returns the selected distance as a rounded float.
    """
    # 1. Compute elementwise absolute differences
    abs_diffs = [abs(xi - yi) for xi, yi in zip(x, y)]
    
    if metric == "euclidean":
        distance = math.sqrt(sum(d**2 for d in abs_diffs))
        
    elif metric == "manhattan":
        distance = sum(abs_diffs)
        
    elif metric == "chebyshev":
        distance = max(abs_diffs)
        
    elif metric == "minkowski":
        distance = sum(d**p for d in abs_diffs) ** (1 / p)
        
    elif metric == "cosine":
        # Compute dot product and vector norms
        dot_product = sum(xi * yi for xi, yi in zip(x, y))
        norm_x = math.sqrt(sum(xi**2 for xi in x))
        norm_y = math.sqrt(sum(yi**2 for yi in y))
        
        # Apply the stated zero-vector cosine rule
        if norm_x == 0 or norm_y == 0:
            return 0.0
            
        distance = 1.0 - (dot_product / (norm_x * norm_y))
        
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    # Return rounded to 4 decimal places
    return round(float(distance), 4)
