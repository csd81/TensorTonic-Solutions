import math

def perplexity(prob_distributions: list, actual_tokens: list) -> float:
    """
    Returns the sequence perplexity.
    """
    n = len(actual_tokens)
    neg_log_probs = [
        -math.log(dist[token])
        for dist, token in zip(prob_distributions, actual_tokens)
    ]
    cross_entropy = sum(neg_log_probs) / n
    return round(float(math.exp(cross_entropy)), 4)
