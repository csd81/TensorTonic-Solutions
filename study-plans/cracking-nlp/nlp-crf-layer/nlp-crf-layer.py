def crf_layer(emissions: list, tag_names: list) -> list:
    """
    Returns one selected tag-name string per position using independent emission decoding.
    """
    selected_tags = []
    
    for row in emissions:
        # Find the index of the highest score. 
        # Python's max() with enumerate preserves the smallest index automatically on a tie.
        max_idx = max(range(len(row)), key=lambda idx: row[idx])
        selected_tags.append(tag_names[max_idx])
        
    return selected_tags
