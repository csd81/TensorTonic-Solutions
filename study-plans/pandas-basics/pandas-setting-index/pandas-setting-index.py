import pandas as pd

def set_index_column(data: dict, index_col: str) -> dict:
    """
    Returns index_values and columns as lists, plus data as a dictionary of lists.
    """
    # 1. Convert the dictionary into a DataFrame to easily handle pandas indexing methods
    df = pd.DataFrame(data)
    
    # 2. Extract the values of the requested column to form the index values list
    index_values = df[index_col].tolist()
    
    # 3. Drop the index column from the DataFrame to leave only the remaining data columns
    df_remaining = df.drop(columns=[index_col])
    
    # 4. Extract the remaining column names as a list
    columns_list = df_remaining.columns.tolist()
    
    # 5. Convert the remaining DataFrame back to a dictionary of lists (orient='list')
    data_dict = df_remaining.to_dict(orient='list')
    
    # 6. Return the strictly formatted dictionary
    return {
        "index_values": index_values,
        "columns": columns_list,
        "data": data_dict
    }
