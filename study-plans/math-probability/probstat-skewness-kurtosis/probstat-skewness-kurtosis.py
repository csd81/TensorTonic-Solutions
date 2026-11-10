import numpy as np

def skewness_kurtosis(data: list) -> dict:
    """
    Returns adjusted statistics and their interpretations in a dictionary.
    """
    n = len(data)
    x = np.array(data, dtype=float)
    
    # 1. Compute mean and sample standard deviation (ddof=1 for n - 1 denominator)
    mean = np.mean(x)
    s = np.std(x, ddof=1)
    
    # 2. Compute the standardized sums
    sum_z3 = np.sum(((x - mean) / s) ** 3)
    sum_z4 = np.sum(((x - mean) / s) ** 4)
    
    # 3. Compute adjusted Fisher sample skewness (g1)
    g1_factor = n / ((n - 1) * (n - 2))
    g1 = g1_factor * sum_z3
    
    # 4. Compute adjusted Fisher excess kurtosis (g2)
    g2_factor1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    g2_factor2 = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    g2 = (g2_factor1 * sum_z4) - g2_factor2
    
    # 5. Round to four decimal places BEFORE classification
    g1_rounded = round(g1, 4)
    g2_rounded = round(g2, 4)
    
    # 6. Classify Skewness
    if g1_rounded > 0.5:
        skew_interpretation = "right-skewed"
    elif g1_rounded < -0.5:
        skew_interpretation = "left-skewed"
    else:
        skew_interpretation = "approximately symmetric"
        
    # 7. Classify Kurtosis
    if g2_rounded > 1.0:
        kurtosis_interpretation = "leptokurtic"
    elif g2_rounded < -1.0:
        kurtosis_interpretation = "platykurtic"
    else:
        kurtosis_interpretation = "mesokurtic"
        
    return {
        "skewness": g1_rounded,
        "kurtosis": g2_rounded,
        "skew_interpretation": skew_interpretation,
        "kurtosis_interpretation": kurtosis_interpretation
    }
