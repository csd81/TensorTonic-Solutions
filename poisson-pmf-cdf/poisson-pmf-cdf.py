import math


def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Base probability for i = 0 events: P(X = 0) = e^(-lam)
    prob = math.exp(-lam)
    cdf = prob

    # Iteratively update probability using P(X = i) = P(X = i - 1) * (lam / i)
    for i in range(1, k + 1):
        prob *= lam / i
        cdf += prob

    return {"pmf": float(prob), "cdf": float(cdf)}
