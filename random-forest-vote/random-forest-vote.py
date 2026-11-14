from collections import Counter

def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    """
    result = []
    
    # zip(*predictions) groups the predictions column-by-column (per sample)
    for sample_votes in zip(*predictions):
        counts = Counter(sample_votes)
        # Sort by (-count, class_label) to prioritize highest votes, then smallest label
        winner = min(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        result.append(int(winner))
        
    return result
