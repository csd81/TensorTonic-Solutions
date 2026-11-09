import re

def extract_patterns(text: str, pattern_type: str) -> list:
    """
    Returns a list of matched strings based on the specified text processing rule.
    """
    patterns = {
        "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        "urls": r'https?://\S+?(?=[,.)!?;:]*(?:\s|$))',
        "dates": r'\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b',
        "money": r'\$\d+(?:\.\d{2})?',
        "hashtags": r'#\w+'
    }
    
    if pattern_type not in patterns:
        return []
        
    return re.findall(patterns[pattern_type], text)
