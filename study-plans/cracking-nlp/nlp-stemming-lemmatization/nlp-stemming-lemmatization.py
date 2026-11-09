def stem_and_lemmatize(word: str, lemma_dict: dict) -> dict:
    """
    Returns a dictionary with string values stem and lemma.
    """
    # 1. Determine the Stem
    stem = word
    
    # Rule 1: Remove 'ing' if at least three characters remain.
    if word.endswith("ing") and len(word) - 3 >= 3:
        stem = word[:-3]
    # Rule 2: Remove 'ly' if at least three characters remain.
    elif word.endswith("ly") and len(word) - 2 >= 3:
        stem = word[:-2]
    # Rule 3: Remove 'ed' if at least three characters remain.
    elif word.endswith("ed") and len(word) - 2 >= 3:
        stem = word[:-2]
    # Rule 4: Remove 's' unless the word ends in 'ss', and only if at least two characters remain.
    elif word.endswith("s") and not word.endswith("ss") and len(word) - 1 >= 2:
        stem = word[:-1]
    # Rule 5: Remove 'er' if at least three characters remain.
    elif word.endswith("er") and len(word) - 2 >= 3:
        stem = word[:-2]
    # Rule 6: Replace 'tion' with 'te' if the result has at least three characters.
    elif word.endswith("tion") and (len(word) - 4 + 2) >= 3:
        stem = word[:-4] + "te"

    # 2. Determine the Lemma
    lemma = lemma_dict.get(word, word)
    
    return {"stem": stem, "lemma": lemma}
