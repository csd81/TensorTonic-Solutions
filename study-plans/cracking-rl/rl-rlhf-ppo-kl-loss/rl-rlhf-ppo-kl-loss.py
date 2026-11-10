import math

def rlhf_ppo_kl_loss(
    log_probs_new: list, 
    log_probs_old: list, 
    log_probs_ref: list, 
    advantages: list, 
    clip_eps: float, 
    kl_coef: float
) -> float:
    """
    Returns the combined PPO and reference-penalty loss, rounded to four decimals.
    """
    T = len(log_probs_new)
    L_clip_list = []
    K_list = []
    
    for t in range(T):
        l_new = log_probs_new[t]
        l_old = log_probs_old[t]
        l_ref = log_probs_ref[t]
        A = advantages[t]
        
        # Probability ratio r_t = exp(l_new - l_old)
        r_t = math.exp(l_new - l_old)
        
        # Clipped ratio
        clipped_r_t = min(max(r_t, 1.0 - clip_eps), 1.0 + clip_eps)
        
        # Unclipped vs clipped objective terms
        term1 = r_t * A
        term2 = clipped_r_t * A
        L_clip_t = min(term1, term2)
        L_clip_list.append(L_clip_t)
        
        # Log-ratio against reference policy
        K_t = l_new - l_ref
        K_list.append(K_t)
        
    # PPO clipped loss is negative average of min terms
    L_clip = -sum(L_clip_list) / T
    # Mean sampled log-ratio penalty term
    K = sum(K_list) / T
    # Total combined loss
    L = L_clip + kl_coef * K
    
    return round(float(L), 4)
