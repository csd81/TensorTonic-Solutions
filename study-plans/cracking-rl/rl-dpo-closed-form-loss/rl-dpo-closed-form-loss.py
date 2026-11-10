import math

def dpo_loss(log_pi_new_w: list, log_pi_new_l: list, log_pi_ref_w: list, log_pi_ref_l: list, beta: float) -> float:
    """
    Returns the mean DPO loss as a float rounded to four decimals.
    Handles extreme margin magnitudes safely without overflow.
    """
    total_loss = 0.0
    N = len(log_pi_new_w)
    
    for i in range(N):
        # Calculate the reference-adjusted margin z_i
        z_i = beta * ((log_pi_new_w[i] - log_pi_ref_w[i]) - (log_pi_new_l[i] - log_pi_ref_l[i]))
        
        # We need to compute -log(sigmoid(z_i)) = log(1 + exp(-z_i))
        # To avoid overflow from exp(-z_i) when -z_i is very large, use the log-sum-exp stabilization:
        neg_z = -z_i
        if neg_z > 700:
            loss_i = neg_z
        elif neg_z < -700:
            loss_i = 0.0
        else:
            loss_i = math.log(1.0 + math.exp(neg_z))
            
        total_loss += loss_i
        
    return round(total_loss / N, 4)
