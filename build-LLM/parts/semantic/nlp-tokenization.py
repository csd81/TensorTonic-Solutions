import re

def tokenize(text: str) -> list:
    """
    Returns a list of token strings.
    """
    # [a-zA-Z0-9_]+ matches consecutive alphanumeric characters and underscores
    # [^a-zA-Z0-9_\s] matches any single character that is not alphanumeric, underscore, or whitespace
    pattern = r'[a-zA-Z0-9_]+|[^a-zA-Z0-9_\s]'
    
    return re.findall(pattern, text)
