import pandas as pd

def merge_dataframes(left: dict, right: dict, on: str, how: str) -> dict:
    """
    Returns a dictionary of merged column lists, with NaN for unmatched cells.
    """
    # Convert input dictionaries to Pandas DataFrames
    df_left = pd.DataFrame(left)
    df_right = pd.DataFrame(right)
    
    # Merge the dataframes based on the provided key and strategy
    merged_df = pd.merge(df_left, df_right, on=on, how=how)
    
    # Convert back to a dictionary of column lists, keeping NaN values intact
    return merged_df.to_dict(orient='list')
