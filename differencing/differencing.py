def differencing(series: list, order: int) -> list:
  """
  Returns the series after the requested differencing order.
  """
  curr = series
  for _ in range(order):
    curr = [curr[i] - curr[i - 1] for i in range(1, len(curr))]
  return curr
