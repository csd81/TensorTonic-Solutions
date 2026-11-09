import pandas as pd

def multi_agg(data: dict, group_col: str, value_col: str, funcs: list) -> dict:
    """
    Returns a dictionary from function names to dictionaries of group aggregates.
    """
    # 1. Convert the input dictionary into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # 2. Group by the specified column and aggregate the value column with the list of functions
    aggregated_df = df.groupby(group_col)[value_col].agg(funcs)
    
    # 3. Convert the resulting DataFrame into the required nested dictionary format
    return aggregated_df.to_dict()

