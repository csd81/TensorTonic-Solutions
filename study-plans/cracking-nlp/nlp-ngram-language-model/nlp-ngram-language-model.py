import numpy as np
from collections import defaultdict

def ngram_language_model(corpus: list, n: int, sentences: list) -> list:
    """
    Returns one float log score per test sentence, rounded to four decimals.
    """
    START = "<s>"
    END = "</s>"
    
    # Define N-gram contexts and full sequences counts
    # Context count: C(h)
    # Full N-gram count: C(h, w)
    context_counts = defaultdict(int)
    ngram_counts = defaultdict(int)
    
    # Unique vocabulary tracks tokens seen in training + END token (excluding START tokens)
    vocab = set()
    total_unigram_tokens = 0
    
    # Process training corpus
    for sentence in corpus:
        # Prepend (n-1) start tokens and append 1 end token
        padded_sent = [START] * (n - 1) + sentence + [END]
        
        # Populate vocabulary and update unigram count total
        for token in sentence + [END]:
            vocab.add(token)
            
        total_unigram_tokens += len(sentence) + 1  # tokens + END token
        
        # Extract N-grams from the training sentence
        for i in range(n - 1, len(padded_sent)):
            w = padded_sent[i]
            h = tuple(padded_sent[i - n + 1 : i])
            
            ngram_counts[(h, w)] += 1
            context_counts[h] += 1

    V_size = len(vocab)
    results = []
    
    # Process test sentences
    for sentence in sentences:
        # Prepend (n-1) start tokens and append 1 end token
        padded_test = [START] * (n - 1) + sentence + [END]
        log_score = 0.0
        
        # Score every original token and the END token
        for i in range(n - 1, len(padded_test)):
            w = padded_test[i]
            h = tuple(padded_test[i - n + 1 : i])
            
            if n == 1:
                # For unigrams, context is empty
                # C(h) is the total number of training tokens including end tokens
                c_h = total_unigram_tokens
                c_hw = ngram_counts[(h, w)]
            else:
                c_h = context_counts[h]
                c_hw = ngram_counts[(h, w)]
            
            # Apply add-one smoothing formula
            prob = (c_hw + 1) / (c_h + V_size)
            log_score += np.log(prob)
            
        results.append(round(float(log_score), 4))
        
    return results
