from scipy.stats import norm

def normal_distribution(mu: float, sigma: float, x: float) -> dict:
    """
    Returns the z-score, CDF, PDF, and one-standard-deviation probability.
    """
    z_score = (x-mu)/sigma
    cdf_val = norm.cdf(x, loc=mu,scale=sigma)
    pdf_val = norm.pdf(x, loc=mu,scale=sigma)
    prob_within_1_std=norm.cdf(mu+sigma, loc=mu,scale=sigma) - norm.cdf(mu-sigma, loc=mu,scale=sigma)
# Return the dictionary with all metrics rounded to 4 decimal places
    return {
        "z_score": round(z_score, 4),
        "cdf": round(cdf_val, 4),
        "pdf": round(pdf_val, 4),
        "prob_within_1_std": round(prob_within_1_std, 4)
    }
