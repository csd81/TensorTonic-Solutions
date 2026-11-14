def deduplicate(records: list, key_columns: list, strategy: str) -> list:
    """
    Returns a list of selected records.
    """
    selected = {}
    
    for record in records:
        key = tuple(record[col] for col in key_columns)
        
        if key not in selected:
            selected[key] = record
        elif strategy == "last":
            selected[key] = record
        elif strategy == "most_complete":
            current_none_count = sum(1 for v in selected[key].values() if v is None)
            new_none_count = sum(1 for v in record.values() if v is None)
            
            # Strictly less than (<) ensures ties are broken by first occurrence
            if new_none_count < current_none_count:
                selected[key] = record
                
    return list(selected.values())
