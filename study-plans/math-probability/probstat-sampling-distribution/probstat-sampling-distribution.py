from math import sqrt
from scipy.stats import norm

def sampling_distribution(mu: float, sigma: float, n: int, threshold: float) -> dict:
    """
    Returns the sampling mean, sampling standard deviation, and lower-tail probability.
    """
    # 1. The mean of the sampling distribution is equal to the population mean
    sampling_mean = mu
    
    # 2. The standard deviation of the sampling distribution (Standard Error)
    sampling_std = sigma / sqrt(n)
    
    # 3. Calculate the cumulative probability using the standard normal CDF
    prob_below_threshold = norm.cdf(threshold, loc=sampling_mean, scale=sampling_std)
    
    # 4. Return the results in a dictionary rounded to 4 decimal places
    return {
        "prob_below_threshold": round(prob_below_threshold, 4),
        "sampling_mean": round(sampling_mean, 4),
        "sampling_std": round(sampling_std, 4)
    }
