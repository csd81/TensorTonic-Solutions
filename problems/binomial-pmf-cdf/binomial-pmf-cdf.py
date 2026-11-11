import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    def term(i: int) -> float:
        return math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))

    pmf = float(term(k))
    cdf = float(sum(term(i) for i in range(k + 1)))

    return {
        "pmf": pmf,
        "cdf": cdf,
    }
