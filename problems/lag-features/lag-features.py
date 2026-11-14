def lag_features(series: list, lags: list) -> list:
    """
    Returns the lag feature matrix.
    """
    max_lag = max(lags)
    return [
        [series[t - lag] for lag in lags]
        for t in range(max_lag, len(series))
    ]
