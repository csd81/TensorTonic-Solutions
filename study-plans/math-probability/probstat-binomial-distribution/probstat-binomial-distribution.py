from math import comb

def binomial_distribution(n: int, p: float, threshold: int) -> dict:
    """
    Returns the PMF, mean, variance, and probability at least threshold.
    """
    
    pmf = []
    for k in range(n+1):
        prob = comb(n,k) * (p**k) * ((1-p) ** (n-k))
        pmf.append(round(prob,4))
        
    
    mean = round(n*p,4)
    variance = round(n*p*(1-p),4)
    prob_at_least = round(sum(pmf[threshold:]),4)

    return {
        "mean": mean, 
        "pmf": pmf, 
        "prob_at_least": prob_at_least,
        "variance": variance
    }