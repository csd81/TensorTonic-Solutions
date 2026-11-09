import pandas as pd

def select_columns(data: dict, columns: list) -> dict:
    """
    Returns a dictionary of value lists in the requested column order.
    Repeated names appear once, in first-occurrence order.
    """
    result = {}
    for col in columns:
        if col in data and col not in result:
            result[col] = data[col]
    return result
