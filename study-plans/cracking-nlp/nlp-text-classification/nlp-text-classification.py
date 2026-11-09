import numpy as np

def text_classification(train_docs: list, train_labels: list, test_docs: list, alpha: float = 1.0) -> list:
    """
    Returns one predicted class-label string per test document using Multinomial Naive Bayes.
    """
    # 1. Track global training vocabulary (unique tokens)
    vocab = set()
    for doc in train_docs:
        for word in doc:
            vocab.add(word)
    vocab_size = len(vocab)
    
    # 2. Document counts per class and overall documents
    total_docs = len(train_docs)
    unique_classes = sorted(list(set(train_labels))) # sorted for deterministic tracking
    class_doc_counts = {c: 0 for c in unique_classes}
    for label in train_labels:
        class_doc_counts[label] += 1
        
    # 3. Token counts per class
    word_counts_per_class = {c: {} for c in unique_classes}
    total_tokens_per_class = {c: 0 for c in unique_classes}
    for doc, label in zip(train_docs, train_labels):
        for word in doc:
            word_counts_per_class[label][word] = word_counts_per_class[label].get(word, 0) + 1
            total_tokens_per_class[label] += 1
            
    # 4. Predict for each test document
    predictions = []
    for doc in test_docs:
        best_score = -float('inf')
        best_class = None
        
        for c in unique_classes:
            # Calculate Prior: ln(D_c / D)
            prior = np.log(class_doc_counts[c] / total_docs)
            
            # Calculate Likelihood sum: sum_{w in d} ln(P(w | c))
            likelihood_sum = 0.0
            for word in doc:
                count_w_c = word_counts_per_class[c].get(word, 0)
                # P(w | c) formula with additive smoothing
                p_w_c = (count_w_c + alpha) / (total_tokens_per_class[c] + alpha * vocab_size)
                likelihood_sum += np.log(p_w_c)
                
            score = prior + likelihood_sum
            
            # Restructured tie-breaking block to handle floating point noise correctly
            if best_class is None:
                best_score = score
                best_class = c
            elif np.isclose(score, best_score, atol=1e-9, rtol=1e-9):
                if c < best_class:
                    best_score = score
                    best_class = c
            elif score > best_score:
                best_score = score
                best_class = c
                
        predictions.append(best_class)
        
    return predictions
