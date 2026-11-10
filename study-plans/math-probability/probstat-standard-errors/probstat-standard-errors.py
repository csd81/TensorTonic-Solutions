import numpy as np

def standard_errors(samples: list) -> dict:
    """
    Returns each sample standard error and their mean.
    """
    se_list = []
    
    for sample in samples:
        # Calculate sample size (n)
        n = len(sample)
        
        # Calculate sample standard deviation with 1 degree of freedom (ddof=1)
        s = np.std(sample, ddof=1)
        
        # Calculate standard error
        se = s / np.sqrt(n)
        
        # Round each individual standard error to 4 decimal places
        se_list.append(round(se, 4))
        
    # Calculate the arithmetic mean of the standard errors
    mean_se = np.mean(se_list)
    
    return {
        "mean_se": round(float(mean_se), 4),
        "standard_errors": se_list
    }
