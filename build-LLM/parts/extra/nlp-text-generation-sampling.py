import math

def text_generation_sampling(logits: list, vocab: list, temperature: float, top_k: int) -> dict:
    """
    Returns a dictionary with probabilities (floats) and top_tokens ([token, probability] lists).
    """
    # 1. Apply temperature scaling
    scaled_logits = [z / temperature for z in logits]
    
    # Numerical stability trick: subtract the max logit to prevent OverflowError
    max_scaled = max(scaled_logits)
    exps = [math.exp(sz - max_scaled) for sz in scaled_logits]
    sum_exps = sum(exps)
    
    # 2. Compute full unrounded probabilities
    full_probs = [e / sum_exps for e in exps]
    
    # 3. Create items with their initial positions to handle top_k and stable tie-breaking
    indexed_items = [(full_probs[i], i, vocab[i]) for i in range(len(vocab))]
    
    # Sort primarily by probability descending, and secondarily by original index ascending
    ranked_items = sorted(indexed_items, key=lambda x: (-x[0], x[1]))
    
    # 4. Handle Top-K filtering and renormalization
    if top_k > 0:
        retained_items = ranked_items[:top_k]
        retained_indices = {item[1] for item in retained_items}
        retained_mass = sum(item[0] for item in retained_items)
        
        final_unrounded_probs = []
        for i in range(len(vocab)):
            if i in retained_indices:
                unrenorm_p = full_probs[i]
                final_unrounded_probs.append(unrenorm_p / retained_mass)
            else:
                final_unrounded_probs.append(0.0)
                
        top_tokens_list = []
        for _, orig_idx, token in retained_items:
            renormalized_p = full_probs[orig_idx] / retained_mass
            top_tokens_list.append([token, round(renormalized_p, 4)])
    else:
        final_unrounded_probs = full_probs
        top_tokens_list = [[item[2], round(item[0], 4)] for item in ranked_items]
        
    rounded_probabilities = [round(p, 4) for p in final_unrounded_probs]
    
    return {
        "probabilities": rounded_probabilities,
        "top_tokens": top_tokens_list
    }
