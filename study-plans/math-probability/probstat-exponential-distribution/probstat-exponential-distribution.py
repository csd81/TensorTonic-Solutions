from math import exp

def exponential_distribution(lam: float, t: float) -> dict:
    """
    Returns the PDF, CDF, survival probability, mean, and variance.
    """
    pdf = lam * exp(-lam * t)
    cdf = 1 - exp(-lam * t)
    survival = exp(-lam * t)
    mean = 1 / lam
    variance = 1 / (lam ** 2)
    
    return {
        "cdf": round(cdf, 4),
        "mean": round(mean, 4),
        "pdf": round(pdf, 4),
        "survival": round(survival, 4),
        "variance": round(variance, 4)
    }
