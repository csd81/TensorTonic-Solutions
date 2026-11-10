def categorical_encode(data: list, method: str = "label"):
    """
    Returns label-encoded metadata or one-hot rows.
    """
    # Step 1: Find unique categories and sort them alphabetically
    classes = sorted(list(set(data)))
    
    # Step 2: Create a fast lookup dictionary mapping each class to its index
    class_to_idx = {category: idx for idx, category in enumerate(classes)}
    
    # Handle Label Encoding
    if method == "label":
        encoded = [class_to_idx[item] for item in data]
        return {
            "classes": classes,
            "encoded": encoded
        }
    
    # Handle One-Hot Encoding
    elif method == "onehot":
        num_classes = len(classes)
        onehot_matrix = []
        
        for item in data:
            # Create a row of zeros matching the number of unique classes
            row = [0] * num_classes
            # Set the corresponding class index position to 1
            target_idx = class_to_idx[item]
            row[target_idx] = 1
            onehot_matrix.append(row)
            
        return onehot_matrix
