import pandas as pd

def concat_dataframes(dfs: list) -> list:
    """
    Returns [shape, data], with shape [rows, columns] and data a dictionary of lists.
    """
    # Convert the input dictionaries into pandas DataFrames
    dataframes = [pd.DataFrame(df) for df in dfs]
    
    # Concatenate them vertically (along rows)
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Get the shape [rows, columns]
    shape = list(combined_df.shape)
    
    # Convert the combined DataFrame to a dictionary of lists
    data_dict = combined_df.to_dict(orient='list')
    
    return [shape, data_dict]
