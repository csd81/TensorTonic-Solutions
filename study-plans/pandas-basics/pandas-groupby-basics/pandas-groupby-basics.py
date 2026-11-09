import pandas as pd

def groupby_basics(data: dict, group_col: str, value_col: str) -> dict:
    """
    Returns sum, mean, and count dictionaries keyed by group label.
    """
    # Load the dictionary into a DataFrame
    df = pd.DataFrame(data)
    
    # Group by the specified column and target the value column
    grouped = df.groupby(group_col)[value_col]
    
    # Construct and return the final dictionary format
    return {
        "sum": grouped.sum().to_dict(),
        "mean": grouped.mean().to_dict(),
        "count": grouped.count().to_dict()
    }
