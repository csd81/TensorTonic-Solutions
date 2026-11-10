import numpy as np

def knn_classifier(X_train: list, y_train: list, X_test: list, k: int) -> list:
    # Convert feature matrices to NumPy float64 arrays
    X_tr = np.array(X_train, dtype=np.float64)
    X_te = np.array(X_test, dtype=np.float64)
    y_tr = np.array(y_train, dtype=np.int64)
    
    predictions = []
    
    for test_row in X_te:
        # Calculate Euclidean distances to all training points
        distances = np.sqrt(np.sum((X_tr - test_row) ** 2, axis=1))
        
        # Track original indices to break distance ties
        indices = np.arange(len(distances))
        
        # Sort primarily by distance, secondarily by original training index
        sorted_indices = np.lexsort((indices, distances))
        
        # Get the labels of the top k nearest neighbors
        k_neighbor_labels = y_tr[sorted_indices[:k]]
        
        # Count frequencies of each unique label
        labels, counts = np.unique(k_neighbor_labels, return_counts=True)
        
        # Find the maximum vote count
        max_votes = np.max(counts)
        
        # Identify all labels that share the maximum vote count
        candidate_labels = labels[counts == max_votes]
        
        # Pick the smallest label to break vote ties
        best_label = np.min(candidate_labels)
        predictions.append(int(best_label))
        
    return predictions
