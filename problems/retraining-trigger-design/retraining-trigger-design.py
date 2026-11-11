def retraining_policy(daily_stats: list, config: dict) -> list:
    """
    Returns a list of retraining day numbers.
    """
    drift_threshold = config["drift_threshold"]
    performance_threshold = config["performance_threshold"]
    max_staleness = config["max_staleness"]
    cooldown = config["cooldown"]
    retrain_cost = config["retrain_cost"]
    remaining_budget = config["budget"]

    days_since_retrain = 0
    last_retrain_day = None
    retrain_days = []

    for entry in daily_stats:
        day = entry["day"]
        days_since_retrain += 1

        triggered = (
            entry["drift_score"] > drift_threshold
            or entry["performance"] < performance_threshold
            or days_since_retrain >= max_staleness
        )

        cooldown_ok = (last_retrain_day is None) or (day - last_retrain_day >= cooldown)
        budget_ok = remaining_budget >= retrain_cost

        if triggered and cooldown_ok and budget_ok:
            retrain_days.append(day)
            remaining_budget -= retrain_cost
            last_retrain_day = day
            days_since_retrain = 0

    return retrain_days
