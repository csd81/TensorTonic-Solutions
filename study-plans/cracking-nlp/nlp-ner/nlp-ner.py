def named_entity_recognition(sentences: list, entity_dict: dict) -> list:
    """
    Implements a dictionary-based named-entity tagging baseline.
    For each token, looks up its exact, case-sensitive entry in entity_dict,
    otherwise assigns "O".
    """
    result = []
    
    for sentence in sentences:
        sentence_tags = []
        for token in sentence:
            # Check exact, case-sensitive match in the dictionary
            tag = entity_dict.get(token, "O")
            sentence_tags.append(tag)
        result.append(sentence_tags)
        
    return result
