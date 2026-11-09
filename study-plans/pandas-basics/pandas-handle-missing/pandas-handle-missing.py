import pandas as pd

def handle_missing(data: dict, fill_value: float) -> dict:
    """
    Returns null_counts as a dictionary of integers and cleaned_data as a
    dictionary of lists.
    """
    # Create a DataFrame from the incoming dictionary of lists
    df = pd.DataFrame(data)
    
    # 1. Count the null/missing values for each column and convert to a dictionary
    null_counts = df.isnull().sum().to_dict()
    
    # 2. Fill the missing values with the provided fill_value
    df_filled = df.fillna(fill_value)
    
    # 3. Convert the cleaned DataFrame back into a dictionary of lists
    # 'list' orient creates a dictionary where columns are keys and values are lists
    cleaned_data = df_filled.to_dict(orient='list')
    
    # Return the final nested dictionary format requested
    return {
        "null_counts": null_counts,
        "cleaned_data": cleaned_data
    }
