def bayesian_update(prior_alpha: float, prior_beta: float, successes: int, failures: int) -> dict:
    """
    Returns posterior parameters and prior and posterior means.
    """
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures
    prior_mean = prior_alpha / (prior_alpha+prior_beta)
    posterior_mean = posterior_alpha / (posterior_alpha+posterior_beta)

    return {
        "posterior_alpha": round(posterior_alpha, 4),
        "posterior_beta": round(posterior_beta, 4),
        "prior_mean": round(float(prior_mean), 4),        
        "posterior_mean": round(float(posterior_mean), 4),
    }