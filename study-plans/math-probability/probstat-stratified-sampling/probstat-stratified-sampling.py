from math import floor, sqrt

def stratified_sample(strata_means: list, strata_stds: list, strata_sizes: list, total_sample: int) -> dict:
    """
    Returns exact allocations, the stratified mean, and its standard error.
    """
    # 1. Compute population weights (Wh)
    N = sum(strata_sizes)
    W = [Nh / N for Nh in strata_sizes]
    
    # 2. Compute ideal proportional allocations
    a = [total_sample * Wh for Wh in W]
    
    # 3. Floor ideal allocations and enforce a minimum of 1 per stratum
    allocations = [max(1, floor(ah)) for ah in a]
    
    # 4. Distribute remaining slots by largest fractional remainder
    remaining_slots = total_sample - sum(allocations)
    if remaining_slots > 0:
        # Track (fractional_remainder, stratum_index)
        remainders = [(ah - floor(ah), i) for i, ah in enumerate(a)]
        # Sort by largest remainder descending; break ties with earlier stratum index ascending
        remainders.sort(key=lambda x: (-x[0], x[1]))
        
        for k in range(remaining_slots):
            idx = remainders[k][1]
            allocations[idx] += 1
            
    # 5. Compute stratified mean
    stratified_mean = sum(Wh * mh for Wh, mh in zip(W, strata_means))
    
    # 6. Compute stratified standard error
    se_variance = sum((Wh**2 * (std**2)) / nh for Wh, std, nh in zip(W, strata_stds, allocations))
    stratified_se = sqrt(se_variance)
    
    return {
        "allocations": allocations,
        "stratified_mean": round(stratified_mean, 4),
        "stratified_se": round(stratified_se, 4)
    }
