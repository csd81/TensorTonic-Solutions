import math
from statistics import NormalDist


def ab_test_conversion(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> dict:
    """
    Returns lift, z, p_value, ci_low, ci_high and significant for a two-proportion z-test.
    """
    # Group conversion rates and observed lift
    p_control = control_conversions / control_n
    p_treatment = treatment_conversions / treatment_n
    lift = p_treatment - p_control

    # Pooled rate and pooled standard error for the z-statistic
    p_pooled = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se_pooled = math.sqrt(
        p_pooled * (1.0 - p_pooled) * (1.0 / control_n + 1.0 / treatment_n)
    )

    std_normal = NormalDist()

    if se_pooled == 0.0:
        z = 0.0
        p_value = 1.0
    else:
        z = lift / se_pooled
        # Two-sided p-value: 2 * (1 - Phi(|z|))
        p_value = 2.0 * (1.0 - std_normal.cdf(abs(z)))

    # Unpooled standard error and critical value for the confidence interval
    se_unpooled = math.sqrt(
        (p_control * (1.0 - p_control) / control_n)
        + (p_treatment * (1.0 - p_treatment) / treatment_n)
    )
    z_crit = std_normal.inv_cdf(1.0 - alpha / 2.0)
    margin = z_crit * se_unpooled

    ci_low = lift - margin
    ci_high = lift + margin
    significant = bool(p_value < alpha)

    return {
        "lift": float(lift),
        "z": float(z),
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": significant,
    }
