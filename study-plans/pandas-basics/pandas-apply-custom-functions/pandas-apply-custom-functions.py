import pandas as pd
import numpy as np

def apply_transform(data: dict, column: str, operation: str) -> dict:
    """
    Transforms the specified numeric column using the selected operation,
    and returns a dictionary of lists including the new '_transformed' column.
    """
    # Load into a DataFrame to utilize pandas operations
    df = pd.DataFrame(data)
    transformed_col = f"{column}_transformed"
    
    if operation == "normalize":
        col_min = df[column].min()
        col_max = df[column].max()
        if col_min == col_max:
            df[transformed_col] = np.nan
        else:
            df[transformed_col] = ((df[column] - col_min) / (col_max - col_min)).round(4)
            
    elif operation == "rank":
        # Assign integer positions from 1 for ascending order
        df[transformed_col] = df[column].rank().astype(int)
        
    elif operation == "cumsum":
        df[transformed_col] = df[column].cumsum()
        
    elif operation == "double":
        df[transformed_col] = df[column] * 2

    # Convert the DataFrame back to a dictionary of lists
    return df.to_dict(orient="list")
