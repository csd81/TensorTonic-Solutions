def detect_drift(reference_counts: list, production_counts: list, threshold: float) -> dict:
    """
    Returns a dictionary with score and drift_detected.
    """
    ref_total = sum(reference_counts)
    prod_total = sum(production_counts)
    
    # Normalize counts into probability distributions and compute Total Variation Distance
    tvd = 0.5 * sum(
        abs((ref / ref_total) - (prod / prod_total))
        for ref, prod in zip(reference_counts, production_counts)
    )
    
    return {
        "score": float(tvd),
        "drift_detected": bool(tvd > threshold)
    }
