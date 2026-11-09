import re

def text_normalize(text: str, operations: list) -> str:
    """
    Returns the normalized string.
    """
    result = text
    for op in operations:
        if op == "lowercase":
            result = result.lower()
        elif op == "remove_punctuation":
            # [^\w\s] matches punctuation, but \w includes underscores. 
            # Adding _ explicitly ensures it gets removed alongside punctuation.
            result = re.sub(r'[^\w\s]|_', '', result)
        elif op == "remove_digits":
            result = re.sub(r'\d', '', result)
        elif op == "collapse_whitespace":
            result = re.sub(r'\s+', ' ', result)
        elif op == "strip":
            result = result.strip()
    return result
