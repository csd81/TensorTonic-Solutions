import pandas as pd

def boolean_filter(data: dict, column: str, threshold: float) -> dict:
    """
    Returns filtered_data as a dictionary of lists and count as an integer.
    """
    # Create the DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    # Filter rows where the specified column is strictly greater than the threshold
    filtered_df = df[df[column] > threshold]
    
    # Convert the filtered DataFrame back to a dictionary of lists
    # orient='list' ensures column names map to a list of values
    filtered_data = filtered_df.to_dict(orient='list')
    
    # Get the count of the matching rows
    count = len(filtered_df)
    
    return {
        "filtered_data": filtered_data,
        "count": count
    }
