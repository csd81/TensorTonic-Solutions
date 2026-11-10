import math

def bradley_terry_loss(r_chosen: list, r_rejected: list) -> float:
    """
    Returns the mean preference loss as a float rounded to four decimals.
    Handles score differences of magnitude 1000 without overflow.
    """
    n = len(r_chosen)
    total_loss = 0.0
    
    for w, l in zip(r_chosen, r_rejected):
        diff = w - l
        
        # Numerically stable log-sigmoid implementation
        if diff < 0:
            log_sig = diff - math.log1p(math.exp(diff))
        else:
            log_sig = -math.log1p(math.exp(-diff))
            
        total_loss += log_sig
        
    mean_loss = -total_loss / n
    return round(mean_loss, 4)
