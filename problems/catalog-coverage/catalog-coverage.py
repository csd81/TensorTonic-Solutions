def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    if n_items == 0:
        return 0.0
    
    unique_items = set()
    for rec_list in recommendations:
        unique_items.update(rec_list)
        
    return len(unique_items) / n_items
