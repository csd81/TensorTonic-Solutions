import numpy as np

def mle_normal(data: list) -> dict:
    """
    Returns the normal-distribution MLEs mu_mle and sigma_mle.
    """

    arr = np.array(data)
    mu_mle =np.mean(arr)
    sigma_mle = np.std(arr,ddof=0)

    
    return {
        "mu_mle": round(float(mu_mle), 4),
        "sigma_mle": round(float(sigma_mle), 4)
    }
