import numpy as np

def bootstrap_ci(data: list, n_bootstraps: int, confidence: float, seed: int) -> list:
    """
    Returns the bootstrap mean and percentile confidence interval.
    """
    # Initialize the reproducible pseudo-random number generator
    rng = np.random.RandomState(seed)
    
    # Convert data into a NumPy array for performance and easy indexing
    data_arr = np.array(data)
    n = len(data_arr)
    
    # Generate all resamples at once with replacement
    resamples = rng.choice(data_arr, size=(n_bootstraps, n), replace=True)
    
    # Compute the mean across the rows (each bootstrap sample)
    boot_means = np.mean(resamples, axis=1)
    
    # Calculate overall average of all bootstrap sample means
    mean_of_means = np.mean(boot_means)
    
    # Determine significance level and percentiles
    alpha = 1.0 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Compute the percentile boundaries using NumPy
    lower_endpoint = np.percentile(boot_means, lower_percentile)
    upper_endpoint = np.percentile(boot_means, upper_percentile)
    
    # Return values rounded to 4 decimal places as requested
    return [
        round(mean_of_means, 4),
        round(lower_endpoint, 4),
        round(upper_endpoint, 4)
    ]
