import math

def ucb1_scores(Q: list, N: list, t: int, c: float) -> list:
    """
    Returns one UCB score per arm, rounded to four decimals.
    """
    scores = []
    for q, n in zip(Q, N):
        # Calculate the exploration bonus: c * sqrt(ln(t) / N(a))
        bonus = c * math.sqrt(math.log(t) / n)
        # Combine the mean reward and the exploration bonus
        ucb_score = q + bonus
        # Round to four decimal places and append to the list
        scores.append(round(ucb_score, 4))
        
    return scores
