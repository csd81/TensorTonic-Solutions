from collections import defaultdict

def pos_tagging(train: list, test: list, default_tag: str = 'NN') -> list:
    """
    Returns one list of tag strings per test sentence.
    """
    # Step 1: Count occurrences of each tag for every word
    word_tag_counts = defaultdict(lambda: defaultdict(int))
    for sentence in train:
        for word, tag in sentence:
            word_tag_counts[word][tag] += 1
            
    # Step 2: Determine the most frequent tag for each word
    most_frequent_tag = {}
    for word, tag_counts in word_tag_counts.items():
        # Sort by count (descending) and then by tag string order (ascending) for ties
        best_tag = min(tag_counts.keys(), key=lambda t: (-tag_counts[t], t))
        most_frequent_tag[word] = best_tag
        
    # Step 3: Tag the test sentences
    result = []
    for sentence in test:
        tagged_sentence = []
        for word in sentence:
            # Assign the most frequent tag if trained, otherwise use the default_tag
            tagged_sentence.append(most_frequent_tag.get(word, default_tag))
        result.append(tagged_sentence)
        
    return result
