import pandas as pd

def head_tail(data: dict, n: int) -> dict:
    """
    Returns head and tail as dictionaries mapping columns to value lists.
    """
    # 1. Create the DataFrame from the input dictionary
    df = pd.DataFrame(data)
    
    # 2. Extract the first n rows and last n rows
    head_df = df.head(n)
    tail_df = df.tail(n)
    
    # 3. Convert both back to dictionaries mapping column names to lists
    return {
        "head": head_df.to_dict(orient="list"),
        "tail": tail_df.to_dict(orient="list")
    }
