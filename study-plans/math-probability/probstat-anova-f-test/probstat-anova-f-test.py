import numpy as np
from scipy.stats import f

def anova_f_test(groups: list) -> list:
    """
    Returns the F-statistic, degrees of freedom, p-value, and decision.
    """
    # 1. Calculate counts and total N
    k = len(groups)
    group_sizes = [len(g) for g in groups]
    N = sum(group_sizes)
    
    # Flatten all data for grand mean calculation
    all_data = [x for g in groups for x in g]
    grand_mean = np.mean(all_data)
    
    # 2. Calculate Group Means and Sum of Squares
    ss_between = 0.0
    ss_within = 0.0
    
    for g in groups:
        g_mean = np.mean(g)
        n_j = len(g)
        
        # SS_between = sum( n_j * (group_mean - grand_mean)^2 )
        ss_between += n_j * ((g_mean - grand_mean) ** 2)
        
        # SS_within = sum( (x_ij - group_mean)^2 )
        ss_within += sum((x - g_mean) ** 2 for x in g)
        
    # 3. Degrees of Freedom
    df_between = k - 1
    df_within = N - k
    
    # 4. Mean Squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    # 5. F-Statistic and p-value
    f_stat = ms_between / ms_within
    p_val = f.sf(f_stat, df_between, df_within)  # Survival function (1 - cdf)
    
    # 6. Evaluation against significance level alpha = 0.05
    reject_h0 = bool(p_val < 0.05)
    
    # Round numeric values to 4 decimal places
    return [
        round(float(f_stat), 4),
        int(df_between),
        int(df_within),
        round(float(p_val), 4),
        reject_h0
    ]
