def seasonal_average(series: list, period: int) -> list:
    """
    Returns the average for each position in the seasonal cycle.
    """
    averages = []
    for p in range(period):
        season_values = series[p::period]
        averages.append(sum(season_values) / len(season_values))
    return averages
