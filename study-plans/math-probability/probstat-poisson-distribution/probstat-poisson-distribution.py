from math import exp, factorial

def poisson_distribution(lam: float, max_k: int) -> list:
    """
    Returns the PMF list, cumulative probability, and zero-event probability.
    """
    pmf_list = []
    for k in range(max_k+1):
        prob = (lam**k)*exp(-lam)/factorial(k)
        pmf_list.append(round(prob,4))
        
    cumulative_prob = round(sum(pmf_list),4)
    p_zero = pmf_list[0]

    return[
        pmf_list,
        cumulative_prob,
        p_zero
    ]
